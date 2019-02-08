#! /usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2018, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import contextlib
import itertools
import re
import os.path
from pathlib import Path
from collections import OrderedDict, namedtuple

from lisa.env import TestEnv, TargetConf
from lisa.platforms.platinfo import PlatformInfo
from lisa.utils import HideExekallID, Loggable, ArtifactPath, get_subclasses, groupby, Serializable
from lisa.conf import MultiSrcConf
from lisa.tests.base import TestBundle, ResultBundle
from lisa.tests.scheduler.load_tracking import FreqInvarianceItem
from lisa.regression import compute_regressions

from exekall.utils import get_name
from exekall.engine import ExprData, Consumer, PrebuiltOperator
from exekall.customization import AdaptorBase

class NonReusable:
    pass

class ExekallArtifactPath(ArtifactPath, NonReusable):
    @classmethod
    def from_expr_data(cls, data:ExprData, consumer:Consumer) -> 'ExekallArtifactPath':
        """
        Factory used when running under `exekall`
        """
        artifact_dir = Path(data['expr_artifact_dir']).resolve()
        consumer_name = get_name(consumer)

        # Find a non-used directory
        for i in itertools.count(1):
            artifact_dir_ = Path(artifact_dir, consumer_name, str(i))
            if not artifact_dir_.exists():
                artifact_dir = artifact_dir_
                break

        cls.get_logger().info('Creating {consumer} artifact storage: {path}'.format(
            consumer = consumer_name,
            path = artifact_dir
        ))
        artifact_dir.mkdir(parents=True)
        # Get canonical absolute paths
        artifact_dir = artifact_dir.resolve()
        root = data['artifact_dir']
        relative = artifact_dir.relative_to(root)
        return cls(root, relative)

class LISAAdaptor(AdaptorBase):
    name = 'LISA'

    def get_non_reusable_type_set(self):
        return {NonReusable}

    def get_prebuilt_set(self):
        non_reusable_type_set = self.get_non_reusable_type_set()
        op_set = set()

        # Try to build as many configurations instances from all the files we
        # are given
        conf_cls_set = set(get_subclasses(MultiSrcConf))
        conf_list = []
        for conf_path in self.args.conf:
            for conf_cls in conf_cls_set:
                try:
                    conf = conf_cls.from_yaml_map(conf_path)
                except KeyError:
                    continue
                else:
                    conf_list.append((conf, conf_path))

        def keyfunc(conf_and_path):
            cls = type(conf_and_path[0])
            # We use the ID since classes are not comparable
            return id(cls), cls

        # Then aggregate all the conf from each type, so they just act as
        # alternative sources.
        for (_, conf_cls), conf_and_path_seq in groupby(conf_list, key=keyfunc):
            conf_and_path_list = list(conf_and_path_seq)
            # Since we use reversed order, we get the source override from the
            # last one.
            conf_and_path_list.reverse()

            conf = conf_and_path_list[0][0]
            for conf_src, conf_path in conf_and_path_list[1:]:
                conf.add_src(conf_path, conf_src, fallback=True)

            op_set.add(PrebuiltOperator(
                conf_cls, [conf],
                non_reusable_type_set=non_reusable_type_set
            ))

        # Inject serialized objects as root operators
        for path in self.args.inject:
            obj = Serializable.from_path(path)
            op_set.add(PrebuiltOperator(type(obj), [obj],
                non_reusable_type_set=non_reusable_type_set
            ))

        return op_set

    def get_hidden_op_set(self, op_set):
        hidden_op_set = {
            op for op in op_set
            if issubclass(op.value_type, HideExekallID)
        }
        self.hidden_op_set = hidden_op_set
        return hidden_op_set

    @staticmethod
    def register_run_param(parser):
        parser.add_argument('--conf', action='append',
            default=[],
            help="LISA configuration file. If multiple configurations of a given type are found, they are merged (last one can override keys in previous ones)")

        parser.add_argument('--inject', action='append',
            metavar='SERIALIZED_OBJECT_PATH',
            default=[],
            help="Serialized object to inject when building expressions")

    @staticmethod
    def register_compare_param(parser):
        parser.add_argument('--alpha', type=float,
            default=5,
            help="""Alpha risk for Fisher exact test in percents.""")

        parser.add_argument('--non-significant', action='store_true',
            help="""Also show non-significant changes of failure rate.""")

        parser.add_argument('--remove-tag', action='append',
            default=[],
            help="""Remove the given tags in the testcase IDs before comparison.""")

    def compare_db_list(self, db_list):
        alpha = self.args.alpha / 100
        show_non_significant = self.args.non_significant

        result_list_old, result_list_new = [
            db.get_roots()
            for db in db_list
        ]

        regr_list = compute_regressions(
            result_list_old,
            result_list_new,
            remove_tags=self.args.remove_tag,
            alpha=alpha,
        )

        if not regr_list:
            print('No matching test IDs have been found, use "--remove-tag board" to match across "board" tags')
            return

        print('testcase failure rate changes with alpha={}\n'.format(alpha))

        id_len = max(len(regr.testcase_id) for regr in regr_list)

        header = '{id:<{id_len}}   old%   new% delta%      pvalue{regr_column}'.format(
            id='testcase'.format(alpha),
            id_len=id_len,
            regr_column=' changed' if show_non_significant else ''
        )
        print(header + '\n' + '-' * len(header))
        for regr in regr_list:
            if regr.significant or show_non_significant:
                old_pc, new_pc = regr.failure_pc
                print('{id:<{id_len}} {old_pc:>5.1f}% {new_pc:>5.1f}% {delta_pc:>5.1f}%    {pval:.2e} {has_regr}'.format(
                    id=regr.testcase_id,
                    old_pc=old_pc,
                    new_pc=new_pc,
                    delta_pc=regr.failure_delta_pc,
                    pval=regr.p_val,
                    id_len=id_len,
                    has_regr='*' if regr.significant and show_non_significant else '',
                ))

    @staticmethod
    def get_default_type_goal_pattern_set():
        return {'*.ResultBundle'}

    @classmethod
    def reload_db(cls, db, path=None):
        # If path is not known, we cannot do anything here
        if not path:
            return db

        # This will relocate ArtifactPath instances to the new absolute path of
        # the results folder, in case it has been moved to another place
        artifact_dir = Path(path).parent.resolve()

        # Relocate ArtifactPath embeded in objects so they will always
        # contain an absolute path that adapts to the local filesystem
        for serial in db.get_all():
            val = serial.value
            try:
                dct = val.__dict__
            except AttributeError:
                continue
            for attr, attr_val in dct.items():
                if isinstance(attr_val, ArtifactPath):
                    new_path = attr_val.with_root(artifact_dir)
                    # Only update paths to existing files, otherwise assume it
                    # was pointing outside the artifact_dir and therefore
                    # should not be fixed up
                    if os.path.exists(new_path):
                        setattr(val, attr, new_path)

        return db

    def finalize_expr(self, expr):
        expr_artifact_dir = expr.data['expr_artifact_dir']
        artifact_dir = expr.data['artifact_dir']
        for expr_val in expr.get_all_vals():
            self._finalize_expr_val(expr_val, artifact_dir, expr_artifact_dir)

    def _finalize_expr_val(self, expr_val, artifact_dir, expr_artifact_dir):
        val = expr_val.value

        def needs_rewriting(val):
            # Only rewrite ArtifactPath path values
            if not isinstance(val, ArtifactPath):
                return False
            # And only if they are a subfolder of artifact_dir. Otherwise, they
            # are something pointing outside of the artifact area, which we
            # cannot handle.
            return artifact_dir.resolve() in Path(val).resolve().parents

        # Add symlinks to artifact folders for ExprValue that were used in the
        # ExprValue graph, but were initially computed for another Expression
        if needs_rewriting(val):
            val = Path(val)
            is_subfolder = (expr_artifact_dir.resolve() in val.resolve().parents)
            # The folder is reachable from our ExprValue, but is not a
            # subfolder of the expr_artifact_dir, so we want to get a
            # symlink to it
            if not is_subfolder:
                # We get the name of the callable
                callable_folder = val.parts[-2]
                folder = expr_artifact_dir/callable_folder

                # We build a relative path back in the hierarchy to the root of
                # all artifacts
                relative_artifact_dir = Path(os.path.relpath(str(artifact_dir), start=str(folder)))

                # The target needs to be a relative symlink, so we replace the
                # absolute artifact_dir by a relative version of it
                target = relative_artifact_dir/val.relative_to(artifact_dir)

                with contextlib.suppress(FileExistsError):
                    folder.mkdir(parents=True)

                for i in itertools.count(1):
                    symlink = Path(folder, str(i))
                    if not symlink.exists():
                        break

                symlink.symlink_to(target, target_is_directory=True)

        for param, param_expr_val in expr_val.param_map.items():
            self._finalize_expr_val(param_expr_val, artifact_dir, expr_artifact_dir)

    @classmethod
    def get_tags(cls, value):
        tags = {}
        if isinstance(value, TestEnv):
            tags['board'] = value.target_conf.get('name')
        elif isinstance(value, PlatformInfo):
            tags['board'] = value.get('name')
        elif isinstance(value, TestBundle):
            tags['board'] = value.plat_info.get('name')
            if isinstance(value, FreqInvarianceItem):
                if value.cpu is not None:
                    tags['cpu'] = '{}@{}'.format(value.cpu, value.freq)
        else:
            tags = super().get_tags(value)

        tags = {k: v for k, v in tags.items() if v is not None}

        return tags

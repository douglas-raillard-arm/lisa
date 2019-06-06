
import json
import sys

def cleanup(val):
    # convert null-terminated str comming as a bytearray into a real str
    if isinstance(val, bytearray):
        i = val.find(b'\0')
        return val[:i].decode('utf-8')
    else:
        return val

def trace_begin():
    pass

event_list = []

def trace_unhandled(event_name, context, event_fields_dict):
    entry = {
        key: cleanup(val)
        for key, val in event_fields_dict.items()
    }
    entry['common_event'] = event_name.split('__', 1)[1]
    event_list.append(entry)

def trace_end():
    #  print(len([e for e in event_list if 'sched_wakeup' in e['common_event']]))
    json_path = sys.argv[2]
    with open(json_path, 'w') as f:
        json.dump(event_list, f)
        #  print(event_list)

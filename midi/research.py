from mido import MidiFile, MetaMessage, Message

f = MidiFile('bach_prelude_and_fugue_inc_maj.mid')
print(str(f))

meta_msg_types = set()

msg_types = set()

print('Meta Messages')
for msg in f:
    if isinstance(msg, MetaMessage): 
        meta_msg_types.add(msg.type)
    else:
        msg_types.add(msg.type)
        
print('Meta Types')
for type in meta_msg_types:
    print('\t', type)

print('Regular Types')
for type in msg_types:
    print('\t', type)



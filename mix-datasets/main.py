import json
import os
import pathlib
# with open(os.path.join(pathlib.Path().resolve(),'..','data-females.json')) as user_file:
#   feman_json = json.load(user_file)
with open('E:\\Work\\University\\PR\\Acoustic_prosody_Example\\data-females.json') as user_file:
   feman_json = json.load(user_file)
with open('E:\\Work\\University\\PR\\Acoustic_prosody_Example\\data-males.json') as user_file:
  man_json = json.load(user_file)
  
result=[]
keys=['F0_std','F0_mean','Voiced_rate','Voiced_duration','Voiced_rPVI','Voiced_nPVI','Voiced_dGPI','Pause_rate','Pause_duration','Speech_rate','SPL_mean','SPL_std']
for json_person in feman_json:
    person={}
    for key in keys:
        person.update({key:json_person[key]})
    for i in range(len(json_person['XArt'][0])):
        person.update({f"XArt_{i}":json_person['XArt'][0][i]})
    person.update({'label':'woman'})
    result.append(person)
for json_person in man_json:
    person={}
    for key in keys:
        person.update({key:json_person[key]})
    for i in range(len(json_person['XArt'][0])):
        person.update({f"XArt_{i}":json_person['XArt'][0][i]})
    person.update({'label':'man'})
    result.append(person)

with open(f"data-mixed.json",'w') as f:
    json.dump(result,f)   
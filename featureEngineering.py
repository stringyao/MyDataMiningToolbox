from itertools import combinations

ralate_var = ['是否经常逛商场的人', '是否去过高档商场', '当月是否看电影', '当月是否景点游览', '当月是否体育场馆消费']

print('waiting for group pair features...')

for rv in combinations(ralate_var, 2):
    rv2 = '_'.join(rv) 
    data['relate_' + rv2] = data[rv[0]] * data[rv[1]]
    print(rv2 + ' finished!')
    
for rv in combinations(ralate_var, 3):
    rv2 = '_'.join(rv) 
    data['relate_' + rv2] = data[rv[0]] * data[rv[1]] * data[rv[2]]
    print(rv2 + ' finished!')
    
for rv in combinations(ralate_var, 4):
    rv2 = '_'.join(rv) 
    data['relate_' + rv2] = data[rv[0]] * data[rv[1]] * data[rv[2]] * data[rv[3]]
    print(rv2 + ' finished!')
    

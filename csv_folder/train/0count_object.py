import glob
import os
import numpy as np

count = 0 

object_name = {
		'car': 0,
		'person' : 1,
		# 'cycle' : 2,
		# 'boat' : 3,
		# 'heavy vehicle': 4,
		# 'plane' :5,
		# 'train' : 6
	}
object_count_output = open("object_count_output.txt", "w+")
print ("Name of the file: ", object_count_output.name)
total_object_count = [0 for index in range(2)]
line = object_count_output.writelines( str(object_name)+'\n\n')

for filename in sorted(	glob.glob('*.csv') ):
	print(filename)
	csv_file = open(filename, 'r')
	line = csv_file.readline()
	count = 0
	object_count = [0 for index in range(2)]
	while line:
		count += 1
		stripped = (line.split(','))[-1].replace('\n','').replace('\r','')
		total_object_count[ object_name[stripped]] += 1
		object_count[ object_name[stripped]] += 1
		line = csv_file.readline()
	csv_file.close()
	line = object_count_output.writelines( filename+' : total_object_count : '+ str(count) + ' : '+ str(object_count)+'\n' )
	print(object_count, 'total : ',object_count[0]+object_count[1])
	# if not os.path.isfile(filename+'.txt'):
		# print (filename)
# print(object_name, object_name['car'])

line = object_count_output.writelines( '\n\n\nfinal object count : '+ ' : '+ str(total_object_count)+'\n' )
print( total_object_count)

object_count_output.close()
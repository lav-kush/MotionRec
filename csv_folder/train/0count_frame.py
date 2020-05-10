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
# object_count_output = open("object_count_output.txt", "w+")
# print ("Name of the file: ", object_count_output.name)
total_object_count = [0 for index in range(2)]
# line = object_count_output.writelines( str(object_name)+'\n\n')
total_frame_count = 0

for filename in sorted(	glob.glob('*.csv') ):
	print(filename)
	csv_file = open(filename, 'r')
	line = csv_file.readline()
	count = 0
	object_count = [0 for index in range(2)]
	old_frame = ''
	new_frame=''
	frame_count = 0
	while line:
		if count == 0:
			old_frame = new_frame = line.split(',')[0]
		new_frame = line.split(',')[0]
		if not (new_frame == old_frame):
			# print old_frame, new_frame,'\n'
			old_frame = new_frame
			frame_count += 1
		count += 1
		# print line, new_frame, old_frame
		stripped = (line.split(','))[-1].replace('\n','').replace('\r','')
		# exit()
		# total_object_count[ object_name[stripped]] += 1
		# object_count[ object_name[stripped]] += 1
		line = csv_file.readline()
	csv_file.close()
	total_frame_count += frame_count
	# line = object_count_output.writelines( filename+' : total_object_count : '+ str(count) + ' : '+ str(object_count)+'\n' )
	print(frame_count)
	# if not os.path.isfile(filename+'.txt'):
		# print (filename)
# print(object_name, object_name['car'])

# line = object_count_output.writelines( '\n\n\nfinal object count : '+ ' : '+ str(total_object_count)+'\n' )
print( total_frame_count)

# object_count_output.close()
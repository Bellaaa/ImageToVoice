import numpy as np
import os
import csv

"""Usage: txt2list(path, # of videos wanted of one person)"""

def txt2list(datadir, n):
	paths = []
	i = 0
	ids = []
	for root, directories, txtnames in os.walk(datadir):

		"""
		root should be the path to the files
		directories should be empty
		txtnames should be the name
		"""
		# for name in txtnames:
		a = root.split('/')
		if len(a) == 3:
			id = a[1]
			content = a[2]
			ids.append([id, content])
	ids = reduce2n(ids, n)
	information = [['id', 'content', 'start', 'end', 'x', 'y', 'w', 'h']]
	for id, c, cnt in ids:
		try:
			path = datadir + '/' + id + '/' + c + '/' + cnt
			a = np.loadtxt(path, dtype=str, delimiter='/t')
			f, x, y, w, h = a[6].split()
			f_e, x_e, y_e, w_e, h_e = a[-1].split()
			information.append([id, c, f, f_e, x, y, w, h])
		except OSError as err:
			pass
	
	with open('ids.csv', 'w', newline = '') as f:
		writer = csv.writer(f)
		writer.writerows(information)


def reduce2n(idlist, n):
	new_ids = []
	cnt = 0
	prev, _ = idlist[0]
	for i, c in idlist:
		curr = i
		if curr == prev:
			cnt += 1
			if cnt >= n:
				prev = curr
				pass
			else:
				prev = curr
				new_ids.append([i, c, '0000' + str(cnt) + '.txt'])
		else:
			cnt = 1
			new_ids.append([i, c, '0000' + str(cnt) + '.txt'])
			prev = curr
	return new_ids

txt2list('txt2', 3)






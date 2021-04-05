def contigency_matrix(actual, predicted):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for i in range(len(actual)):
		if actual[i] == '0' and predicted[i] == '0':
			tn += 1
		elif actual[i] == '1' and predicted[i] == '1':
			tp += 1
		elif actual[i] == '0' and predicted[i] == '1':
			fp += 1
		elif actual[i] == '1' and predicted[i] == '0':
			fn += 1
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	accuracy = (tp+tn)/(tp+fn+fp+tn)
	F1=(2*precision*recall)/(precision+recall)
	return (tp,tn,fp,fn,precision,recall,accuracy,F1)

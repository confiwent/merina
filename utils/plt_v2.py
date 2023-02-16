import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

RESULTS_FOLDER_FCC = '../Results/test/lin/fcc/'
RESULTS_FOLDER_OBE = '../Results/test/lin/oboe/'
RESULTS_FOLDER_3GP = '../Results/test/lin/3gp/'
RESULTS_FOLDER_PUF = '../Results/test/lin/puffer/'
RESULTS_FOLDER_PUF2 = '../Results/test/lin/puffer2/'
RESULTS_FOLDER_FH = '../Results/test/lin/fh/'

RESULTS_FOLDER_FCC_LOG = '../Results/test/log/fcc/'
RESULTS_FOLDER_OBE_LOG = '../Results/test/log/oboe/'
RESULTS_FOLDER_3GP_LOG = '../Results/test/log/3gp/'
RESULTS_FOLDER_PUF_LOG = '../Results/test/log/puffer/'
RESULTS_FOLDER_FH_LOG = '../Results/test/log/fh/'
RESULTS_FOLDER_PUF2_LOG = '../Results/test/log/puffer2/'

PIC_FOLDER = '../Results/pic/'
NUM_BINS = 200
BITS_IN_BYTE = 8.0
INIT_CHUNK = 4
M_IN_K = 1000.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 49
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
K_IN_M = 1000.0
REBUF_P_LIN = 4.3
REBUF_P_LOG = 2.66
SMOOTH_P = 1
COLOR_MAP = plt.cm.rainbow#plt.cm.jet #nipy_spectral, Set1,Paired plt.cm.rainbow#
METRIC = 'log'
NORM = False #True 
PROPOSED_SCHEME = 'test_mpc'
PROPOSED_SCHEME_NAME = 'RobustMPC'
# SCHEMES = ['test_cmc', 'test_merina', 'test_bola'] # 
# METHOD_LABEL = ['Comyco', 'MERINA', 'BOLA'] # 'Penseive' 'BOLA', 'RobustMPC', 'BOLA', , 'Comyco'
LINE_STY = ['--', ':', '-.', '--', ':', '-.', '-', '-']


parser = argparse.ArgumentParser(description='PLOT RESULTS')
parser.add_argument('--log', action='store_true', help='Use logarithmic form QoE metric')
parser.add_argument('--tf', action='store_true', help='Use FCC traces')
parser.add_argument('--tfh', action='store_true', help='Use FCCand3GP traces')
parser.add_argument('--t3g', action='store_true', help='Use 3GP traces')
parser.add_argument('--to', action='store_true', help='Use Oboe traces')
parser.add_argument('--tp', action='store_true', help='Use Puffer traces')
parser.add_argument('--tp2', action='store_true', help='Use Puffer2 traces')
parser.add_argument('--comyco', action='store_true', help='Show the results of Comyco')
parser.add_argument('--mpc', action='store_true', help='Show the results of RobustMPC')
parser.add_argument('--pensieve', action='store_true', help='Show the results of Penseive')
parser.add_argument('--imrl', action='store_true', help='Show the results of MERINA')
parser.add_argument('--bola', action='store_true', help='Show the results of BOLA')
parser.add_argument('--adp', action='store_true', help='Show the results of adaptation')
parser.add_argument('--fugu', action='store_true', help='Show the results of FUGU')
parser.add_argument('--bayes', action='store_true', help='Show the results of BayesMPC')


def save_csv(data, file_name):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(file_name,index=False,sep=',')

def main():

	args = parser.parse_args()
	# args = parser.parse_args(args = ['--tf', '--merina', '--bola', '--mpc'])
	if args.tf:
		results_folder = RESULTS_FOLDER_FCC_LOG if args.log else RESULTS_FOLDER_FCC
		save_folder = 'log/fcc/' if args.log else 'fcc/'
	elif args.t3g:
		results_folder = RESULTS_FOLDER_3GP_LOG if args.log else RESULTS_FOLDER_3GP
		save_folder = 'log/3gp/' if args.log else '3gp/'
	elif args.tfh:
		results_folder = RESULTS_FOLDER_FH_LOG if args.log else RESULTS_FOLDER_FH
		save_folder = 'log/fh/' if args.log else 'fh/'
	elif args.to:
		results_folder = RESULTS_FOLDER_OBE_LOG if args.log else RESULTS_FOLDER_OBE
		save_folder = 'log/oboe/' if args.log else 'oboe/'
	elif args.tp:
		results_folder = RESULTS_FOLDER_PUF_LOG if args.log else RESULTS_FOLDER_PUF
		save_folder = 'log/puffer/' if args.log else 'puffer/'
	elif args.tp2:
		results_folder = RESULTS_FOLDER_PUF2_LOG if args.log else RESULTS_FOLDER_PUF2
		save_folder = 'log/puffer2/' if args.log else 'puffer2/'
	else:
		print("Please choose the throughput data traces!!!")

	schemes_show = []
	schemes_label = []

	if args.bola:
		schemes_show.append('test_bola')
		schemes_label.append('BOLA')
	if args.mpc:
		schemes_show.append('test_mpc')
		schemes_label.append('RobustMPC')
	if args.pensieve:
		schemes_show.append('test_a3c')
		schemes_label.append('Pensieve')
	if args.comyco:
		schemes_show.append('test_cmc')
		schemes_label.append('Comyco')
	if args.fugu:
		schemes_show.append('test_fugu')
		schemes_label.append('Fugu')
	if args.bayes:
		schemes_show.append('test_bayes')
		schemes_label.append('BayesMPC')
	if args.imrl:
		schemes_show.append('test_merina')
		schemes_label.append('MERINA')


	qoe_metric = METRIC
	normalize = NORM
	norm_addition = 1.2
	time_all = {}
	bit_rate_all = {}
	buff_all = {}
	rebuf_all = {}
	bw_all = {}
	raw_reward_all = {}

	for scheme in schemes_show:
		time_all[scheme] = {}
		raw_reward_all[scheme] = {}
		bit_rate_all[scheme] = {}
		buff_all[scheme] = {}
		rebuf_all[scheme] = {}
		bw_all[scheme] = {}

	log_files = os.listdir(results_folder)
	for log_file in log_files:

		time_ms = []
		bit_rate = []
		buff = []
		rebuf = []
		bw = []
		reward = []

		print(log_file)

		with open(results_folder + log_file, 'rb') as f:
			for line in f:
				parse = line.split()
				if len(parse) <= 1:
					break
				time_ms.append(float(parse[0]))
				bit_rate.append(int(parse[1]))
				buff.append(float(parse[2]))
				rebuf.append(float(parse[3]))
				bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
				reward.append(float(parse[6]))
		
		time_ms = np.array(time_ms)
		time_ms -= time_ms[0]
		
		# print log_file

		for scheme in schemes_show:
			if scheme in log_file:
				time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
				bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
				buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
				rebuf_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = rebuf
				bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
				raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
				break

	# ---- ---- ---- ----
	# Reward records
	# ---- ---- ---- ----
		
	log_file_all = []
	reward_all = {}
	reward_quality = {}
	reward_rebuf = {}
	reward_smooth = {}
	reward_improvement = {}
	rebuf_improvement = {}
	for scheme in schemes_show:
		reward_all[scheme] = []
		reward_quality[scheme] = []
		reward_rebuf[scheme] = []
		reward_smooth[scheme] = []
		if scheme != PROPOSED_SCHEME:
			reward_improvement[scheme]= []
			rebuf_improvement[scheme] = []

	for l in time_all[schemes_show[0]]:
		schemes_check = True
		for scheme in schemes_show:
			if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
				schemes_check = False
				break
		if schemes_check:
			log_file_all.append(l)
			for scheme in schemes_show:
				## record the total QoE data
				reward_all[scheme].append(np.mean(raw_reward_all[scheme][l][INIT_CHUNK:VIDEO_LEN]))
				##--------------------record the individual terms in QoE -------------------------
				## caculate the average video quality and quality smoothness penalty
				if qoe_metric == 'lin':
					bit_rate_ = bit_rate_all[scheme][l][INIT_CHUNK:VIDEO_LEN]
					last_bit_rate_ = np.roll(bit_rate_, 1)
					last_bit_rate_[0] = bit_rate_all[scheme][l][INIT_CHUNK - 1]
					trace_q = [i / M_IN_K for i in bit_rate_]
					trace_s = [SMOOTH_P * np.abs(bit_rate_[idx] - last_bit_rate_[idx]) / M_IN_K for idx in range(len(bit_rate_))]
					if normalize:
						reward_quality[scheme].append(np.log(norm_addition+np.mean(trace_q)))
						reward_smooth[scheme].append(np.log(norm_addition+np.mean(trace_s)))
					else:
						reward_quality[scheme].append(np.mean(trace_q))
						reward_smooth[scheme].append(np.mean(trace_s))
					## set the rebuffer penalty
					rebuf_p = REBUF_P_LIN
				else:
					bit_rate_ = bit_rate_all[scheme][l][INIT_CHUNK:VIDEO_LEN]
					last_bit_rate_ = np.roll(bit_rate_, 1)
					last_bit_rate_[0] = bit_rate_all[scheme][l][INIT_CHUNK - 1]
					trace_q = [np.log(i / float(VIDEO_BIT_RATE[0])) for i in bit_rate_]
					trace_q_last = [np.log(i / float(VIDEO_BIT_RATE[0])) for i in last_bit_rate_]
					trace_s = [SMOOTH_P * np.abs(trace_q[idx] - trace_q_last[idx]) for idx in range(len(trace_q))]
					if normalize:
						reward_quality[scheme].append(np.log(norm_addition+np.mean(trace_q)))
						reward_smooth[scheme].append(np.log(norm_addition+np.mean(trace_s)))
					else:
						reward_quality[scheme].append(np.mean(trace_q))
						reward_smooth[scheme].append(np.mean(trace_s))
					## set the rebuffer penalty
					rebuf_p = REBUF_P_LOG

				## caculate the average rebuffer penalty
				trace_r = rebuf_all[scheme][l][INIT_CHUNK:VIDEO_LEN]
				trace_r = [rebuf_p * element for element in trace_r]
				if normalize:
					reward_rebuf[scheme].append(np.log(norm_addition+np.mean(trace_r)))
				else:
					reward_rebuf[scheme].append(np.mean(trace_r))
				## ------------------------------------------------------------------------------

	## ------------------ calculate the reward improvement---------------------------
	for l in range(len(reward_all[PROPOSED_SCHEME])):
		comparison_schemes = [schemes_show[i] for i in range(len(schemes_show))]
		comparison_schemes.remove(PROPOSED_SCHEME)
		for scheme in comparison_schemes:
			reward_improvement[scheme].append(-min(float(reward_all[PROPOSED_SCHEME][l] - reward_all[scheme][l]), 10.0)) # abs(reward_all[scheme][l])
			rebuf_improvement[scheme].append(-max(float(reward_rebuf[PROPOSED_SCHEME][l] - reward_rebuf[scheme][l]), -20))

	## calculate the average QoE and individual terms (mean value + std)
	mean_rewards = {}
	mean_QoE = []
	std_QoE = []
	mean_quality = []
	std_quality = []
	mean_rebuf = []
	std_rebuf = []
	mean_smooth = []
	std_smooth = []
	for scheme in schemes_show:
		mean_rewards[scheme] = np.mean(reward_all[scheme])
		mean_QoE.append(np.mean(reward_all[scheme]))
		std_QoE.append(np.std(reward_all[scheme]))
		##----------------mean value and std------------------
		mean_quality.append(np.mean(reward_quality[scheme]))
		std_quality.append(np.std(reward_quality[scheme]))
		# smoothness penalty 
		mean_smooth.append(np.mean(reward_smooth[scheme]))
		std_smooth.append(np.std(reward_smooth[scheme]))
		# rebuffer penalty
		mean_rebuf.append(np.mean(reward_rebuf[scheme]))
		std_rebuf.append(np.std(reward_rebuf[scheme]))

	## ------------------------------------load the data into texts------------------------------
	results_load = {}
	results_load['LABEL'] = schemes_label
	results_load['QoE'] = mean_QoE
	results_load['QoE_std'] = std_QoE
	results_load['q_mean'] = mean_quality
	results_load['q_std'] = std_quality
	results_load['r_mean'] = mean_rebuf
	results_load['r_std'] = std_rebuf
	results_load['s_mean'] = mean_smooth
	results_load['s_std'] = std_smooth

	save_file = './log_results/data_log.csv' if args.log else './log_results/data_lin.csv'
	save_csv(results_load, save_file)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	for scheme in schemes_show:
		ax.plot(reward_all[scheme])
	
	SCHEMES_REW = []
	# for scheme in schemes_show:
	# 	SCHEMES_REW.append(scheme + ': ' + str('%.3f' % mean_rewards[scheme]))
	for idx in range(len(schemes_show)):
		SCHEMES_REW.append(schemes_label[idx] + ': ' + str('%.3f' % mean_rewards[schemes_show[idx]]))
		# SCHEMES_REW.append(schemes_label[idx])

	colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		j.set_color(colors[i])

	ax.legend(SCHEMES_REW, loc=4)
	
	plt.ylabel('total reward')
	plt.xlabel('trace index')
	# if not os.path.exists(PIC_FOLDER + save_folder):
	# 	os.mkdir(PIC_FOLDER + save_folder)
	# plt.savefig(PIC_FOLDER + save_folder + "avg_QoE.pdf")
	plt.show()

	# ---- ---- ---- ----
	# CDF 
	# ---- ---- ---- ----
	SCHEMES_REW = []
	for idx in range(len(schemes_show)):
		# SCHEMES_REW.append(schemes_label[idx] + ': ' + str('%.3f' % mean_rewards[schemes_show[idx]]))
		SCHEMES_REW.append(schemes_label[idx])

	fig = plt.figure()
	ax = fig.add_subplot(111)

	
	for scheme in schemes_show:
		cdf_values = {}
		values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
		cumulative = np.cumsum(values)/float(len(reward_all[scheme]))
		cumulative = np.insert(cumulative, 0, 0)
		# ax.plot(base[:-1], cumulative)
		# cdf_values[scheme] = {}
		cdf_values['value'] = base
		cdf_values['cumulative'] = cumulative
		cdf_data_frame = pd.DataFrame(cdf_values)
		sns.lineplot(x="value", y="cumulative", data=cdf_data_frame)
	# cdf_data_frame = pd.pivot
		

	# colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		# j.set_color(colors[i])
		plt.setp(j, linestyle = LINE_STY[i], linewidth = 2.6)
	# sns.lineplot(x=)
	# sns.lineplot(x="value", y="cumulative", data=cdf_data_frame)

	legend = ax.legend(SCHEMES_REW, fontsize = 18)
	frame = legend.get_frame()
	frame.set_alpha(0)
	frame.set_facecolor('none')
	
	plt.ylabel('CDF (Perc. of sessions)', fontsize = 20)
	if args.log:
		plt.xlabel("Average Values of Chunk's $QoE_{log}$", fontsize = 20)
	else:
		plt.xlabel("Average Values of Chunk's $QoE_{lin}$", fontsize = 20)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_linewidth(2.8)
	ax.spines['left'].set_linewidth(2.8)
	# plt.savefig(PIC_FOLDER + save_folder + "CDF_QoE.pdf")
	# plt.title('HSDPA and FCC') # HSDPA , FCC , Oboe
	plt.show()

	#################################################################
	# QoE reward_improvement
	#################################################################

	fig = plt.figure()
	ax = fig.add_subplot(111)

	for scheme in comparison_schemes:
		cdf_values = {}
		values, base = np.histogram(reward_improvement[scheme], bins=NUM_BINS)
		cumulative = np.cumsum(values)/float(len(reward_improvement[scheme]))
		cumulative = np.insert(cumulative, 0, 0)
		# ax.plot(base[:-1], cumulative)	
		cdf_values['value'] = base[:]
		cdf_values['cumulative'] = cumulative
		cdf_data_frame = pd.DataFrame(cdf_values)
		sns.lineplot(x="value", y="cumulative", data=cdf_data_frame)

	# colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		# j.set_color(colors[i])	
		plt.setp(j, linestyle = LINE_STY[i+1], linewidth = 2.6) #, marker = HATCH[i]

	comparison_schemes_names = [schemes_label[i] for i in range(len(schemes_label))]
	comparison_schemes_names.remove(PROPOSED_SCHEME_NAME)

	legend = ax.legend(comparison_schemes_names, loc='best', fontsize = 18)
		# legend = ax.legend(SCHEMES_REW, loc=4, fontsize = 14)
	frame = legend.get_frame()
	frame.set_alpha(0)
	frame.set_facecolor('none')
	plt.ylabel('CDF (Perc. of sessions)', fontsize = 20)
	if args.log:
		plt.xlabel("Avg. $QoE_{log}$ improvement", fontsize = 20)
	else:
		plt.xlabel("Avg. $QoE_{lin}$ improvement", fontsize = 20)
	# plt.xlabel("Avg. QoE improvement", fontsize = 18)
	plt.xticks(fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.ylim([0.0,1.0])
	# plt.xlim(-0.2, 1)
	plt.vlines(0, 0, 1, colors='k',linestyles='solid')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_linewidth(2.5)
	ax.spines['left'].set_linewidth(2.5)
	# plt.title('HSDPA and FCC') # HSDPA , FCC , Oboe
	# plt.grid()
	# plt.savefig(PIC_FOLDER + save_folder + "CDF_QoE_IM.pdf")
	plt.show()


if __name__ == '__main__':
	main()

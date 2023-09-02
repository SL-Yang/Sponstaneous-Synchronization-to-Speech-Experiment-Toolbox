from expy import *
import sys
import imp
import pickle
import random

imp.reload(sys)
#remember to start Praat for recording!!!!
start(fullscreen=1,mouse_visible=True,sample_rate=16000,background_color=C_black)


#load soundsa
volume_adj_sound = loadSound('sounds/adjustment.wav')
example_sound = loadSound('sounds/example_gc.wav')
exp_sound = loadSound('sounds/main.wav')


#Instruction-understand experimental devices
drawPic('instructions/welcome.png',w=1500,h=900)
show(3)
clear()
drawPic('instructions/continue_button.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})
clear()
drawPic('instructions/volume_button.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})
clear()


#Instruction-prepare for the experiment
drawPic('instructions/prepare_1.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})
clear()
drawPic('instructions/prepare_2.png',w=1500,h=900)
playSound(example_sound,busy=True)
waitForResponse(allowed_keys={key_.A:'a'})
clear() 
drawPic('instructions/prepare_3.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})
clear() 


#Preparation-adjust the volume
drawPic('instructions/volume_adj_1.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})  
clear()

repeat=True
while repeat == True:
        drawPic('instructions/volume_adj_2.png',w=1500,h=900)
        playSound(volume_adj_sound)
        key,rt = waitForResponse(allowed_keys={key_.A:'a',key_.G:'b'})
        if key == 'a':
                repeat = False
clear()
textSlide('实验过程中请勿再次调节音量',size=50)
show(3)


#randomly choose the syllable asked after experiment 1
G5 = pickle.load(open('G5.p','rb'))
unknown_syl = pickle.load(open('uk_sy.p','rb'))
known_syl= random.sample(G5,6)
ask_syl = unknown_syl+known_syl




#Instruction-understand waht should be done in experiment 1
drawPic('instructions/exp_1_1.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})
clear()
drawPic('instructions/exp_1_2.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})
clear()
drawPic('instructions/exp_1_3.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})
clear()
drawPic('instructions/exp_1_4.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})
clear()

#Experiment 1
for i in range(3,0,-1):
	textSlide(f'请准备，实验将在{i}秒后开始',size=50)
	show(1)
	clear()

playSound(exp_sound,busy=True)


#Test the memory
drawPic('instructions/recall.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})
clear()
responses = {}
accuracy = 0
for ind,syl in enumerate(ask_syl):
	
	textSlide(f'音节：{syl} 在刚才听到的音频中出现过吗？',size=50)
	resp,rt = waitForResponse(allowed_keys={key_.A:'y',key_.G:'n'})
	clear()
	responses[ind] = [rt,resp,syl in G5]

	if syl in unknown_syl:
		if resp == 'y':
			answer = 'wrong'
			continue
		else:
			answer = 'correct'
			accuracy += 1
			continue
	elif syl in G5:
		if resp == 'y':
			answer = 'correct'
			accuracy += 1
			continue
		else:
			answer = 'wrong'
			continue
	else:
		continue
textSlide(f'您的正确率是：{accuracy/10*100}%',size=50)
show(3)

#Instruction-understand waht should be done in experiment 2
textSlide('最后一个试验',size=50)
show(3)
drawPic('instructions/exp_2_1.png',w=1500,h=900)
waitForResponse(allowed_keys={key_.A:'a'})
clear()



#Experiment 2
for i in range(3,0,-1):
	textSlide(f'请准备，实验将在{i}秒后开始',size=50)
	show(1)
	clear()
playSound(exp_sound,busy=True)


#Over
alertAndQuit('实验结束，谢谢您的配合,请等待我们统计您的数据后领取奖励',size=40)



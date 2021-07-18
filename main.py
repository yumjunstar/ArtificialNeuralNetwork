#MNIST 데이터 기반 인공 신경망

import numpy
import time
import scipy.special


# 신경망 클래스 정의
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.b = 0
        # input, hidden, output 노드 갯수 설정

        self.lr = learningrate  # 학습률 설정
        # self.wih=(numpy.random.rand(self.hnodes,self.inodes)-0.5)
        # self.who=(numpy.random.rand(self.onodes,self.hnodes)-0.5)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 가중치 랜덤하게 설정
        self.activation_function = lambda x: scipy.special.expit(x)
        # 활성화 함수 정의 시그모이드 함수로

    # @jit(nopython=True)
    def train(self, inputs_list, targets_list):
        # 입력 리스트를 2차원의 행렬로 반환
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T  # 실제 목표값

        # 은닉 계층으로 들어오는 신호를 계산
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs  # 오차값
        hidden_errors = numpy.dot(self.who.T, output_errors)  # 은닉 계층 오차값

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def timer(self):
        self.a = time.time()
        self.diff = self.a - self.b
        self.b = self.a
        return self.diff

    def save(self):
        numpy.save("wih", self.wih)
        numpy.save("who", self.who)

    def load(self):
        self.wih = numpy.load("wih.npy")
        self.who = numpy.load("who.npy")



#MNIST 테스팅 함수
from matplotlib import pyplot
#%matplotlib inline
def testing():

    scoreboard=[]
    test_data=open("mnist_test.csv","r")
    data_list=test_data.readlines()
    test_data.close()
    times = len(data_list)
    for i in range(times):
        all_values = data_list[i].split(",")
        inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        result_array = n.query(inputs)
        #print (result_array)
        #result=list(result_array).index(max(list(result_array)))

        result = numpy.argmax(result_array)
        #print("정답",all_values[0],"네트워크의 답 :", result, "확률 :", float(result_array[result]*100), "%")
        if int(result)==int(all_values[0]):
            scoreboard.append(1)
        else:
            scoreboard.append(0)
            """
            image_array = numpy.asfarray(all_values[1:]).reshape(28,28)
            pyplot.imshow(image_array,cmap="Greys",interpolation='None')
            pyplot.show()
            print("정답",all_values[0],"네트워크의 답 :", result, "확률 :", float(result_array[result]*100), "%")
            print (result_array)
            print ("오답")
            """
    print ("테스트 완료! 글씨 인식 정답률 : ",sum(scoreboard)/len(scoreboard)*100,"%")


#손글씨 테스팅 함수
from PIL import Image
def testing_real():
    scoreboard=[]
    for i in range(10):
        loc = str(i)+".png"
        img_array=numpy.array(Image.open(loc).convert('L'))#이미지 읽기 flatten true이면 색상 있는 것을 회색으로
        img_data = 255.0-img_array.reshape(784) #한줄의 행렬로 변환
        img_data = (img_data/255.0*0.98)+0.01 #0~1사이로 변환
        output=n.query(img_data)#신경망에 질문하기

        #img_data_output=img_data.reshape(28,28)#이미지를 다시 28x28행렬로 변환
        #pyplot.imshow(img_data_output,cmap="Greys",interpolation='None')
        #pyplot.show()
        #print (img_data.reshape(28,28))


        #print(output)#각 가중치 확인
        result = numpy.argmax(output)
        #print (result)#값 구하기
        if int(result)==i:
            scoreboard.append(1)
        else:
            scoreboard.append(0)
    print ("테스트 완료! 글씨 인식 정답률 : ",sum(scoreboard)/len(scoreboard)*100,"%")



import time
import sys

def training(ech, training_data_list):

    start=time.time()
    i = 0
    for e in range(ech):#주기
        for record in training_data_list:#각 행별로 record에 넣어짐(행->열 순)
            if (i*perc)%(len(training_data_list)*ech) == 0:#일정 시간 마다 출력 할 수 있도록
                lp=(i)/(len(training_data_list)*ech)
                total_time=n.timer()*perc
                time_passed=(time.time()-start)
                time_remain=total_time-time_passed
                print ("학습 진행도:",lp*100,"%,","지난 시간:",round(time_passed,1),"초,","남은 시간:",round(time_remain,1),"초")
            all_values=record.split(',')
            inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01#asfarray float 리스트->행렬 변환 메소드
            targets = numpy.zeros(output_nodes)+0.01#output_nodes 크기의 0.01인 행렬 (target이 0이면 폭주)
            targets[int(all_values[0])]=0.99#레이블 값에 0.99
            n.train(inputs,targets)
            i+=1
        print (e+1,"주기 학습 완료")
    print ("완전히 학습 완료")


#시작점 여러 데이터를 모으기 위해 수정된 형태
for hidden_c in range(3,11): #은닉 층 노드 배수 갯수
    for ech_c in range(1,6): #주기
        if hidden_c == 3:
            ech_c+=4
            if ech_c>=6:
                break
        input_nodes=784
        hidden_nodes=1000*hidden_c
        output_nodes=10
        learning_rate=0.2
        perc=20#몇퍼센트마다 정보를 알려줄지
        perc=100/perc
        ech=ech_c#주기



        n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)


        training_data_file=open("mnist_train.csv",'r')
        training_data_list=training_data_file.readlines()
        training_data_file.close()

        print ("입력 계층 갯수:",input_nodes)
        print ("은닉 계층 갯수:",hidden_nodes)
        print ("출력 계층 갯수:",output_nodes)
        print ("학습률:", learning_rate)
        print ("학습 반복:",ech)

        print ("초기값 설정 완료")
        #a=int(input("기존 학습데이터를 불러오고 싶다면 1 그대로 진행하고 싶다면 0을 입력해주세요"))
        #if a==1:
        #    try:
        #        n.load()
        #    except Exception as e:
        #        print ("학습 데이터를 불러오는 과정에서 오류가 발생했습니다.",e)
        #    else:
        #        print ("학습 데이터를 성공적으로 불러왔습니다.")
        #        sys.exit(0)

        print ("")

        #신경망 학습 시키기
        training (ech, training_data_list)

        n.save()
        print ("")
        print ("테스트 준비중...")
        testing()
        testing_real()

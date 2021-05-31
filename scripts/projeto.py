#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function, division
import rospy
import numpy as np
import numpy
import tf
import math
import cv2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge, CvBridgeError
from numpy import linalg
from tf import transformations
from tf import TransformerROS
import tf2_ros
from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped

from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import cv2.aruco as aruco
import sys

print("EXECUTE ANTES da 1.a vez: ")
print("wget https://github.com/Insper/robot21.1/raw/main/projeto/ros_projeto/scripts/MobileNetSSD_deploy.caffemodel")
print("PARA TER OS PESOS DA REDE NEURAL")

import visao_module


bridge = CvBridge()

cv_image = None
media = []
centro = []
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos


area = 0.0 # Variavel com a area do maior contorno

# Só usar se os relógios ROS da Raspberry e do Linux desktop estiverem sincronizados. 
# Descarta imagens que chegam atrasadas demais
check_delay = False 

resultados = [] # Criacao de uma variavel global para guardar os resultados vistos

x = 0
y = 0
z = 0 
id = 0

frame = "camera_link"
# frame = "head_camera"  # DESCOMENTE para usar com webcam USB via roslaunch tag_tracking usbcam

tfl = 0

tf_buffer = tf2_ros.Buffer()

aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters  = aruco.DetectorParameters_create()
parameters.minDistanceToBorder = 0
parameters.adaptiveThreshWinSizeMax = 1000
marker_size  = 20
calib_path  = "/home/borg/catkin_ws/src/robot21.1/robot_proj_-paonachapagourmet-3a/scripts/"
camera_matrix   = np.loadtxt(calib_path+'cameraMatrix_raspi.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path+'cameraDistortion_raspi.txt', delimiter=',')

low_yellow = np.array([25, 100, 100],dtype=np.uint8)
high_yellow = np.array([30, 255, 255],dtype=np.uint8)
c_amarelo = [0,0]
centro = [0,0]
def filter_color(bgr, low, high):
	""" REturns a mask within the range"""
	hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, low, high)
	return mask     

# Função centro de massa baseada na aula 02  https://github.com/Insper/robot202/blob/master/aula02/aula02_Exemplos_Adicionais.ipynb
# Esta função calcula centro de massa de máscara binária 0-255 também, não só de contorno
def center_of_mass(mask):
	""" Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
	M = cv2.moments(mask)
	# Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	return [int(cX), int(cY)]

def crosshair(img, point, size, color):
	""" Desenha um crosshair centrado no point.
		point deve ser uma tupla (x,y)
		color é uma tupla R,G,B uint8
	"""
	x,y = point
	cv2.line(img,(x - size,y),(x + size,y),color,5)
	cv2.line(img,(x,y - size),(x, y + size),color,5)






avanca = False
def scaneou(dado):
	leitura = np.array(dado.ranges).round(decimals=2)
	global avanca
	if leitura[0] < 0.40:
		avanca = False
	else:
		avanca = True

bifurca=[999,999]
angulos=[0,0,0,0]
ang_bf=0
posicao_anterior=[999,999]
ang_anterior=0
achou_creeper=False
def recebe_odometria(data):
	global x
	global y
	global state
	global state_cor
	global bifurca
	global angulos
	global ang_bf
	global posicao_anterior
	global ang_anterior
	global achou_creeper

	x = data.pose.pose.position.x
	y = data.pose.pose.position.y

	quat = data.pose.pose.orientation
	lista = [quat.x, quat.y, quat.z, quat.w]

	angulos = np.degrees(transformations.euler_from_quaternion(lista)) 
	angulos[2] = (angulos[2] + 360) % 360


	if state==1 or state==5 or state==11:
		bifurca=[x,y]
		ang_bf=angulos[2]
	if state_cor==1:
		ang_bf=angulos[2]
	if achou_creeper:
		posicao_anterior=[x,y]
		ang_anterior=angulos[2]
		achou_creeper=False

def calcula_distancia (bifurca):
	global x
	global y
	

	x0=bifurca[0]
	y0=bifurca[1]

	return math.sqrt((y-y0)**2+(x-x0)**2)


def verifica_cor (imagem,cor):
	frame_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
	
	if cor=="red":
		cor1 = np.array([0, 50, 50])
		cor2 = np.array([4, 255, 255])
		segmentado_cor = cv2.inRange(frame_hsv, cor1, cor2)

		cor1 = np.array([176, 50, 50])
		cor2 = np.array([180, 255, 255])
		segmentado_cor += cv2.inRange(frame_hsv, cor1, cor2)

		segmentado_cor = cv2.morphologyEx(segmentado_cor,cv2.MORPH_CLOSE,np.ones((7, 7)))

	elif cor=="blue":
		cor1=np.array([80, 50, 50])
		cor2=np.array([95, 255, 255])
		segmentado_cor = cv2.inRange(frame_hsv, cor1, cor2)
		segmentado_cor = cv2.morphologyEx(segmentado_cor,cv2.MORPH_CLOSE,np.ones((7, 7)))

	elif cor=="green":
		cor1=np.array([45, 50, 50])
		cor2=np.array([ 60, 255, 255])
		segmentado_cor = cv2.inRange(frame_hsv, cor1, cor2)
		segmentado_cor = cv2.morphologyEx(segmentado_cor,cv2.MORPH_CLOSE,np.ones((7, 7)))


	return segmentado_cor

def cross(img_rgb, point, color, width,length):
        cv2.line(img_rgb, (point[0] - length/2, point[1]),  (point[0] + length/2, point[1]), color ,width, length)
        cv2.line(img_rgb, (point[0], point[1] - length/2), (point[0], point[1] + length/2),color ,width, length)



maior_contorno = None
estado_anterior=False
distance=999
id_to_find=0
achou=False
ver_cor=True
# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
	global cv_image
	global media
	global centro
	global resultados
	global c_amarelo
	global distance
	global id_to_find
	global achou
	global state
	global maior_contorno
	global achou_creeper
	global estado_anterior
	global state_cor
	global ver_cor
	missao=['green']
	#missao=['red']
	#missao=['blue']
	now = rospy.get_rostime()
	imgtime = imagem.header.stamp
	lag = now-imgtime # calcula o lag
	delay = lag.nsecs
	# print("delay ", "{:.3f}".format(delay/1.0E9))
	if delay > atraso and check_delay==True:
		# Esta logica do delay so' precisa ser usada com robo real e rede wifi 
		# serve para descartar imagens antigas
		print("Descartando por causa do delay do frame:", delay)
		return 
	try:
		temp_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
		centro, saida_net, resultados =  visao_module.processa(temp_image) 



		segmentado_cor=verifica_cor(temp_image,missao[0])

		contornos, arvore = cv2.findContours(segmentado_cor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		maior_contorno_area = 0

		for cnt in contornos:
			area = cv2.contourArea(cnt)
			if area > maior_contorno_area:
				maior_contorno = cnt
				maior_contorno_area = area
		if not maior_contorno is None :
			cv2.drawContours(saida_net, [maior_contorno], -1, [0, 0, 255], 5)
			maior_contorno = np.reshape(maior_contorno, (maior_contorno.shape[0], 2))
			media = maior_contorno.mean(axis=0)
			media = media.astype(np.int32)
			cv2.circle(saida_net, (media[0], media[1]), 5, [0, 255, 0])


		if ver_cor:
			print (cv2.contourArea(maior_contorno))
			if cv2.contourArea(maior_contorno)>=1000:
				achou_creeper=True
				estado_anterior=state
				state='cor'
				ver_cor=False
				state_cor=0
			



		gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
		if ids is not None:
			aruco.drawDetectedMarkers(saida_net, corners, ids) 
			ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
			rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
			if ids[0]==id_to_find:
				achou=True
				distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)



		

		mask_amarelo=filter_color(temp_image,low_yellow,high_yellow)
		#if mask_amarelo.Contours is not None:       
		c_amarelo=center_of_mass(mask_amarelo)
		crosshair(saida_net,c_amarelo,3,(255,0,0))
		# Desnecessário - Hough e MobileNet já abrem janelas
		cv_image = saida_net.copy()
		#cv2.imshow("cv_image", cv_image)
		cv2.imshow("cv_image", cv_image)
		cv2.waitKey(1)
	except CvBridgeError as e:
		print('ex', e)

if __name__=="__main__":
	rospy.init_node("cor")
	distancia_bf=0
	state=0
	state_cor=0

	topico_imagem = "/camera/image/compressed"

	recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
	recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)


	print("Usando ", topico_imagem)

	velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
	ref_odometria = rospy.Subscriber("/odom", Odometry, recebe_odometria)

	tfl = tf2_ros.TransformListener(tf_buffer) #conversao do sistema de coordenadas 
	tolerancia = 25
	v=0

	try:
		# Inicializando - por default gira no sentido anti-horário

		
		while not rospy.is_shutdown():
			if state==0:
			#Anda reto e para de acordo com a distância do id 100.
				id_to_find=100

				if distance<105:
					v-=0.1
					vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
					velocidade_saida.publish(vel)
					if v<=0:
						state=1
						achou=False
						v=0
				else:
					if v<0.5:
						v+=0.05
					if (c_amarelo[0] > centro[0]):
						w=-0.25
					if (c_amarelo[0] < centro[0]):
						w=0.25
					if (abs(c_amarelo[0]- centro[0])<5):
						w=0
					vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
					velocidade_saida.publish(vel)

				
			if state==1:
				#Gira até encontrar o id 50.
				id_to_find = 50
				vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.5))
				velocidade_saida.publish(vel)
				if achou :
					state=2
					achou=False

			if state==2:
				#Anda até que esteja perto do id 50.
				if distance<113:
					v-=0.1
					vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
					velocidade_saida.publish(vel)
					if v<=0:
						state=3
						v=0
				else:
					if v<0.5:
						v+=0.05
					if (c_amarelo[0] > centro[0]):
						w=-0.25
					if (c_amarelo[0] < centro[0]):
						w=0.25
					if (abs(c_amarelo[0]- centro[0])<5):
						w=0
					vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
					velocidade_saida.publish(vel)
			
			if state==3:
				#Gira até que o ângulo seja oposto ao final do estágio 1.
				ang_desejado=270+(ang_bf-90)

				
				if angulos[2]<=ang_desejado+2 and angulos[2]>=ang_desejado-2:
					state=4
				else:
					vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.5))
					velocidade_saida.publish(vel)



			if state==4:
				#Anda até que a posição seja parecida àquela gravada no estágio 1.
				distancia_bf=calcula_distancia(bifurca)
				if distancia_bf<=0.35 and distancia_bf>=0:
					v-=0.1
					vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
					velocidade_saida.publish(vel)
					if v<=0:
						state=5
						achou=False
						v=0
				else:
					if v<0.5:
						v+=0.05
					if (c_amarelo[0] > centro[0]):
						w=-0.25
					if (c_amarelo[0] < centro[0]):
						w=0.25
					if (abs(c_amarelo[0]- centro[0])<5):
						w=0
					vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
					velocidade_saida.publish(vel)

			
			if state==5:
				#Gira até achar o id 150.
				id_to_find = 150
				vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.5))
				velocidade_saida.publish(vel)
				if achou :
					state=6
					achou=False

			if state==6:
				#Anda até que esteja perto do id 150.
				if distance<113:
					v-=0.1
					vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
					velocidade_saida.publish(vel)
					if v<=0:
						state=7
						v=0
				else:
					if v<0.5:
						v+=0.05
					if (c_amarelo[0] > centro[0]):
						w=-0.25
					if (c_amarelo[0] < centro[0]):
						w=0.25
					if (abs(c_amarelo[0]- centro[0])<5):
						w=0
					vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
					velocidade_saida.publish(vel)


			if state==7:
				#Gira até que o ângulo seja oposto ao final do estágio 5.
				ang_desejado=ang_bf-180

				
				if angulos[2]<=ang_desejado+2 and angulos[2]>=ang_desejado-2:
					state=8
				else:
					vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.5))
					velocidade_saida.publish(vel)


			
			if state==8:
				#Anda até que a posição seja parecida àquela gravada no estágio 5.
				distancia_bf=calcula_distancia(bifurca)
				if distancia_bf<=0.35 and distancia_bf>=0:
					v-=0.1
					vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
					velocidade_saida.publish(vel)
					if v<=0:
						state=9
						achou=False
						v=0
				else:
					if v<0.5:
						v+=0.05
					if (c_amarelo[0] > centro[0]):
						w=-0.25
					if (c_amarelo[0] < centro[0]):
						w=0.25
					if (abs(c_amarelo[0]- centro[0])<5):
						w=0
					vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
					velocidade_saida.publish(vel)


			if state==9:
				#Gira até achar o id 200.
				id_to_find = 200
				vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.5))
				velocidade_saida.publish(vel)
				if achou :
					state=10
					achou=False

			if state==10:
				#Anda até que esteja perto do id 200.
				if distance<75:
					v-=0.1
					vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
					velocidade_saida.publish(vel)
					if v<=0:
						state=11
						v=0
				else:
					if v<0.5:
						v+=0.05
					if (c_amarelo[0] > centro[0]):
						w=-0.25
					if (c_amarelo[0] < centro[0]):
						w=0.25
					if (abs(c_amarelo[0]- centro[0])<5):
						w=0
					vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
					velocidade_saida.publish(vel)

			if state==11:
				#Gira até que o ângulo seja perto de 90.
				ang_desejado=90
				start_time = rospy.Time.now()
				
				if angulos[2]<=ang_desejado+2 and angulos[2]>=ang_desejado-2:
					state=12
				else:
					vel = Twist(Vector3(0,0,0), Vector3(0,0,0.5))
					velocidade_saida.publish(vel)

			if state==12:
				#Anda até que a posição seja parecida àquela gravada no estágio 11.
				distancia_bf=calcula_distancia(bifurca)
				if rospy.Time.now() - start_time >= rospy.Duration.from_sec(10):
					print ("Passou o tempo")
					if distancia_bf<=0.4 and distancia_bf>=0:
						v-=0.1
						vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
						velocidade_saida.publish(vel)
						if v<=0:
							state=13
							achou=False
							v=0
					else:
						if v<0.5:
							v+=0.05
						if (c_amarelo[0] > centro[0]):
							w=-0.35
						if (c_amarelo[0] < centro[0]):
							w=0.35
						if (abs(c_amarelo[0]- centro[0])<5):
							w=0
						vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
						velocidade_saida.publish(vel)
				else:
					print ("Não passou o tempo")
					if v<0.5:
						v+=0.05
					if (c_amarelo[0] > centro[0]):
						w=-0.35
					if (c_amarelo[0] < centro[0]):
						w=0.35
					if (abs(c_amarelo[0]- centro[0])<5):
						w=0
					vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
					velocidade_saida.publish(vel)

			if state==13:
				#Gira até achar o id 100.
				id_to_find = 100
				vel = Twist(Vector3(0,0,0), Vector3(0,0,0.5))
				velocidade_saida.publish(vel)
				if achou :
					state=0
					achou=False

			if state=="cor":
				print (cv2.contourArea(maior_contorno))
				if state_cor==0:
					#0-Parar 
					v-=0.1
					vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
					velocidade_saida.publish(vel)
					if v<=0:
						state_cor=1
						identifica=True
						v=0

				if state_cor==1:
					#1-Girar ate centralizar o creeper e andar ate que esteja perto.

					if (abs(media[0]- centro[0])<10):
						identifica=False
						print ('Centralizou')
                
			
                
					if identifica:
						print ('Identificando')
						if not maior_contorno is None :
							if cv2.contourArea(maior_contorno)>=500:
								if (media[0] > centro[0]):
									print ('Direita')
									vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.2))
									velocidade_saida.publish(vel)

								if (media[0] < centro[0]):
									print ('Esquerda')
									vel = Twist(Vector3(0,0,0), Vector3(0,0,0.2))
									velocidade_saida.publish(vel)
							else:
								print ('Contorno menor')
								vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.2))
								velocidade_saida.publish(vel)
					
						else:
							
							vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.2))
							velocidade_saida.publish(vel)

					else:
						print ('Avançar')
						if avanca:
							if not maior_contorno is None :
								if cv2.contourArea(maior_contorno)>=600:
									if (media[0] > centro[0]):
										w=-0.3
									if (media[0] < centro[0]):
										w=0.3
									if (abs(media[0]- centro[0])<10):
										w=0
									vel = Twist(Vector3(0.3,0,0), Vector3(0,0,w))
									velocidade_saida.publish(vel)
								else:
									identifica=True
							else:
								identifica=True
						else:
							v-=0.1
							vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
							velocidade_saida.publish(vel)
							if v<=0:
								state_cor=2
								calculo=True
								v=0

							
					
					

				if state_cor==2:
					#2-girar até que o ângulo seja oposto ao do estado 1.
					if calculo:
						if ang_bf>=0 and ang_bf<=180:
							ang_desejado=ang_bf+180
						elif ang_bf>180 and ang_bf<=360:
							ang_desejado=ang_bf-180
						calculo=False
					else:
						if angulos[2]<=ang_desejado+2 and angulos[2]>=ang_desejado-2:
							state_cor=3
						else:
							vel = Twist(Vector3(0,0,0), Vector3(0,0,0.5))
							velocidade_saida.publish(vel)


				if state_cor==3:
					#3-Andar até a posição_anterior
					distancia_bf=calcula_distancia(posicao_anterior)
					if distancia_bf<=0.6:
						v-=0.1
						vel = Twist(Vector3(v,0,0), Vector3(0,0,0))
						velocidade_saida.publish(vel)
						if v<=0:
							state_cor=4
							v=0
					else:
						if v<0.5:
							v+=0.05
						if (angulos[2] > ang_desejado):
							w=-0.35
						if (angulos[2] < ang_desejado):
							w=0.35
						if angulos[2]<=ang_desejado+2 and angulos[2]>=ang_desejado-2:
							w=0
						vel = Twist(Vector3(v,0,0), Vector3(0,0,w))
						velocidade_saida.publish(vel)
			
					
				if state_cor==4:
					#4-Girar até o ang_anterior.
					if angulos[2]<=ang_anterior+2 and angulos[2]>=ang_anterior-2:
						state=estado_anterior
						ver_cor=False
	
					else:
						vel = Twist(Vector3(0,0,0), Vector3(0,0,0.5))
						velocidade_saida.publish(vel)



			#nada impede do robo voltar ao creeper (talvez resolver por tempo)

				print (state_cor)
						

			
			print (state)

			rospy.sleep(0.1)
			

	except rospy.ROSInterruptException:
		print("Ocorreu uma exceção com o rospy")

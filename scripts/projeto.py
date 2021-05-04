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

low_yellow = np.array([22, 50, 50],dtype=np.uint8)
high_yellow = np.array([36, 255, 255],dtype=np.uint8)
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






distance=999
id_to_find=0

# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
	global cv_image
	global media
	global centro
	global resultados
	global c_amarelo
	global distance
	global id_to_find

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

		gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
		if ids is not None:
			aruco.drawDetectedMarkers(saida_net, corners, ids) 
			ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
			rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]
			if ids[0]==id_to_find:
				distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)



		

		mask_amarelo=filter_color(saida_net,low_yellow,high_yellow)       
		c_amarelo=center_of_mass(mask_amarelo)
		crosshair(saida_net,c_amarelo,3,(255,0,0))
		# Desnecessário - Hough e MobileNet já abrem janelas
		cv_image = saida_net.copy()
		cv2.imshow("cv_image", cv_image)
		cv2.waitKey(1)
	except CvBridgeError as e:
		print('ex', e)
	
if __name__=="__main__":
	rospy.init_node("cor")
	state=0
	topico_imagem = "/camera/image/compressed"

	recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)


	print("Usando ", topico_imagem)

	velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

	tfl = tf2_ros.TransformListener(tf_buffer) #conversao do sistema de coordenadas 
	tolerancia = 25

	try:
		# Inicializando - por default gira no sentido anti-horário

		
		while not rospy.is_shutdown():
			if state==0:
				id_to_find=100

				if distance<105:
					vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
					velocidade_saida.publish(vel)
					#start_time = rospy.Time.to_sec(rospy.Time.now())
					state=1
				else:
					if (c_amarelo[0] > centro[0]):
						w=-0.15
					if (c_amarelo[0] < centro[0]):
						w=0.15
					if (abs(c_amarelo[0]- centro[0])<5):
						w=0
					vel = Twist(Vector3(0.3,0,0), Vector3(0,0,w))
					velocidade_saida.publish(vel)

				
			if state==1:
				vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.2))
				velocidade_saida.publish(vel)
				#if rospy.Time.to_sec(rospy.Time.now()) - start_time > math.pi/(0.2*6):
				state=2

			if state==2:
				if (c_amarelo[0] > centro[0]):
					w=-0.15
				if (c_amarelo[0] < centro[0]):
					w=0.15
				if (abs(c_amarelo[0]- centro[0])<5):
					w=0
				vel = Twist(Vector3(0.3,0,0), Vector3(0,0,w))
				velocidade_saida.publish(vel)

			print (state)
			rospy.sleep(0.1)
			

	except rospy.ROSInterruptException:
		print("Ocorreu uma exceção com o rospy")




	

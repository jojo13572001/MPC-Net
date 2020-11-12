import zmq
import json

class Jzmq:
	  """ Json with zmq to send and receive json data"""

	  def __init__(self, ip, port):
	      """ initialization """
	      context = zmq.Context()
	      self.socket = context.socket(zmq.REQ)
	      address = "tcp://"+ip+":"+str(port)
	      self.socket.connect(address)

	  def sendJson(self, message):
	  	  """ Send Json message """
	  	  message = json.dumps(message)
	  	  self.socket.send_string(message)
	  	  response = self.socket.recv()
	  	  return json.loads(response)


class JzmqServer:
	  def __init__(self, ip, port):
	      """ initialization """
	      context = zmq.Context()
	      self.socket = context.socket(zmq.REP)
	      address = "tcp://"+ip+":"+str(port)
	      self.socket.bind(address)

	  def recvJson(self):
	  	  request = self.socket.recv();
	  	  message = json.loads(request)
	  	  return message

	  def responseJson(self, message):
	  	  message = json.dumps(message)
	  	  self.socket.send_string(message)

print("Start zeromq tcp server ...")
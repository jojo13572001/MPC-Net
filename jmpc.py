import jzmq
import json

class Jmpc:
	  """ Json with zmq to send and receive json data"""

	  def __init__(self, ip="127.0.0.1", port=5678):
	  	  """ initialization """
	  	  self.zmq = jzmq.Jzmq(ip, port)

	  def resetTrajectory(self):
	  	  request = {"id":0,"jsonrpc":"2.0","method":"reset","params":{}}
	  	  return self.zmq.sendJson(request)

	  def getTrajectory(self):
  	  	  request = {"id":0,"jsonrpc":"2.0","method":"get_trajectory","params":{}}
  	  	  return self.zmq.sendJson(request)

	  def computePolicy(self, state, timeStamp):
	  	  request = {"id":0, "jsonrpc":"2.0", "method":"compute_policy", "params":{"state":state,"time":timeStamp}}
	  	  return self.zmq.sendJson(request).get("result")

	  def getControl(self, dt, state, timeStamp):
	  	  request = {"id":0, "jsonrpc":"2.0", "method":"get_control", "params":{"control_period":dt,"state":state,"time":timeStamp}}
	  	  return self.zmq.sendJson(request).get("result")

	  def getNextState(self, control, dt, state, stop=False):
	  	  request = {"id":0, "jsonrpc":"2.0", "method":"get_next_state", "params":{"control":control,"delta_t":dt,"initial_state":state}, "stop":stop}
	  	  return self.zmq.sendJson(request).get("result")
	  	  
	  def getValueFunctionStateDerivative(self, state, timeStamp):
	  	  request = {"id":0, "jsonrpc":"2.0", "method":"get_value_function_state_derivative", "params":{"state":state,"time":timeStamp}}
	  	  return self.zmq.sendJson(request)

	  def setInitState(self, dt, initState, trajectoryLength, count=0):
	  	  request = {"id":0, "jsonrpc":"2.0", "method":"set_init_state", "params":{"initial_state":initState,"delta_t":dt, "trajectoryLength":trajectoryLength, "count":count}}
	  	  return self.zmq.sendJson(request)

	  def setInitTrainingState(self, it, learningIterations):
	  	  request = {"id":0, "jsonrpc":"2.0", "method":"set_init_training_state", "params":{"it":it, "learningIterations":learningIterations}}
	  	  return self.zmq.sendJson(request).get("result").get("firstState")

	  def setState(self, state, index):
	  	  request = {"id":0, "jsonrpc":"2.0", "method":"set_state", "params":{"state":state, "index":index}}
	  	  return self.zmq.sendJson(request).get("result")

class JmpcServer:
	  def __init__(self, ip="127.0.0.1", port=5678):
	  	  """ initialization """
	  	  self.zmq = jzmq.JzmqServer(ip, port)

	  def recvJson(self):
	  	  return self.zmq.recvJson()

	  def recvControl(self):
	  	  message = self.recvJson()
	  	  return message.get("params").get("control"), message.get("stop")

	  def recvInitState(self):
	  	  message = self.recvJson().get("params")
	  	  response = {"id":0, "jsonrpc":"2.0", "result":True}
	  	  self.zmq.responseJson(response)
	  	  return message.get("initial_state"), message.get("delta_t"), message.get("trajectoryLength"), message.get("count")

	  def recvInitTrainingState(self, firstState):
	  	  message = self.recvJson().get("params")
	  	  response = {"id":0, "jsonrpc":"2.0", "result":{"firstState":firstState}}
	  	  self.zmq.responseJson(response)
	  	  return message.get("it"), message.get("learningIterations")

	  def recvState(self):
	  	  message = self.recvJson().get("params")
	  	  response = {"id":0, "jsonrpc":"2.0", "result":True}
	  	  self.zmq.responseJson(response)
	  	  return message.get("state"), message.get("index")

	  def responseCurrentState(self, state):
	  	  request = {"id":0, "jsonrpc":"2.0", "result":state}
	  	  return self.zmq.responseJson(request)

print("Create a MPC service ...")
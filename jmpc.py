import jzmq

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
	  	  return self.zmq.sendJson(request)

	  def getControl(self, dt, state, timeStamp):
	  	  request = {"id":0, "jsonrpc":"2.0", "method":"get_control", "params":{"control_period":dt,"state":state,"time":timeStamp}}
	  	  return self.zmq.sendJson(request)

	  def getNextState(self, control, dt, state):
	  	  request = {"id":0, "jsonrpc":"2.0", "method":"get_next_state", "params":{"control":control,"delta_t":dt,"initial_state":state}}
	  	  return self.zmq.sendJson(request)
	  def getValueFunctionStateDerivative(self, state, timeStamp):
	  	  request = {"id":0, "jsonrpc":"2.0", "method":"get_value_function_state_derivative", "params":{"state":state,"time":timeStamp}}
	  	  return self.zmq.sendJson(request)

print("Create a MPC service ...")
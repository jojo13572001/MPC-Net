# Trajectory Tracking MPC API

## C++ API
```
void set_trajectory(const Vector &times, const Vectors &trajectory);

void reset();

// state = [position, velocity]
bool compute_policy(value_type time, const Vector &state);

Vector get_value_function_state_derivative(value_type time, const Vector &state);
Vector get_constraint_lagrangian(value_type time, const Vector &state);
Vector get_control(value_type time, const Vector &state, value_type control_period);
Vector get_next_state(const Vector &initial_state, const Vector &control, value_type delta_t);
```

## Flow
```
while (true) {
  mpc.set_trajectory(times, trajectory);
  Vector initial_state = ...;

  Vector state = initial_state;
  for (int time = 0; time < final_time; time += dt) {
   if (!mpc.compute_policy(time, state))
      error;
  
    Vector control = mpc.get_control(time, state, dt);
    Vector next_state = mpc.get_next_state(state, control, dt);

    state = next_state;
  }
  mpc.reset()
}
```

## JsonRPC data format

+ assume
  + 3 time points
  + 4 joints
+ state vector includes position(head) and velocity(tail)
 
 
### set_trajectory
+ C++ API
`void set_trajectory(const Vector &times, const Vectors &trajectory)`

+ request
```
{
"id":0,
"jsonrpc":"2.0",
"method":"set_trajectory",
"params":{"times":[0.0,1.0,2.0],"trajectory":[[101.0,102.0,103.0,104.0],[201.0,202.0,203.0,204.0],[301.0,302.0,303.0,304.0]]}
}
```

+ response
```
{
"id":0,
"jsonrpc":"2.0",
"result":null
}
```


### get_trajectory
+ C++ API
`void get_trajectory(Vector &times, Vectors &trajectory)`

+ request
```
{
"id":0,
"jsonrpc":"2.0",
"method":"get_trajectory",
"params":{}
}
```

+ response
```
{
"id":0,
"jsonrpc":"2.0",
"result":{"times":[1.0,2.0,3.0],"trajectory":[[0.1,0.2,0.3,0.4,0.5,0.6],[1.1,1.2,1.3,1.4,1.5,1.6],[2.1,2.2,2.3,2.4,2.5,2.6]]}
}
```


### reset
+ C++ API
`void reset()`

+ request
```
{
"id":0,
"jsonrpc":"2.0",
"method":"reset",
"params":{}
}
```

+ response
```
{
"id":0,
"jsonrpc":"2.0",
"result":null
}
```

### compute_policy
+ C++ API
`bool compute_policy(value_type time, const Vector &state)`

+ request
```
{
"id":0,
"jsonrpc":"2.0",
"method":"compute_policy",
"params":{"state":[1.1,2.2,3.3,4.4,-1.1,-2.2,-3.3,-4.4],"time":1.1}
}
```

+ response
```
response = 
{
"id":0,
"jsonrpc":"2.0",
"result":true
}
```

### get_control
+ C++ API
`Vector get_control(value_type time, const Vector &state, value_type control_period)`

+ request
```
{
"id":0,
"jsonrpc":"2.0",
"method":"get_control",
"params":{"control_period":0.03,"state":[1.1,2.2,3.3,4.4,-1.1,-2.2,-3.3,-4.4],"time":2.2}
}
```

+ response
```
{
"id":0,
"jsonrpc":"2.0",
"result":[1.0,2.0,3.0,4.0]
}
```

### get_next_state
+ C++ API
`Vector get_next_state(const Vector &initial_state, const Vector &control, value_type delta_t)`

+ request
```
{
"id":0,
"jsonrpc":"2.0",
"method":"get_next_state",
"params":{"control":[19.0,18.0,17.0,16.0],"delta_t":0.03,"initial_state":[1.1,2.2,3.3,4.4,-1.1,-2.2,-3.3,-4.4]}
}
```

+ response
```
{
"id":0,
"jsonrpc":"2.0",
"result":[3.0,2.0,1.0,0.0,-1.0,-2.0,-3.0,-4.0]
}
```


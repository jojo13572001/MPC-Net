## Pick and Place MPC App

### command options
```
--pick arg          Pick coordinate x,y,z (default: -0.3,0.3,0.05)
--place arg         Place coordinate x,y,z (default: -0.3,-0.3,0.05)
--thruz arg         z coordinate of horizon path (default: 0.35)
--dt arg            Sample Time (dt) (default: 0.03)
--wait arg          number of dt to wait at place (default: 5)
```

### show trajectory in Bullet
```
--urdf arg          URDF file
--trajectory        for trajectory (default: 0)
```

### verify MPC computation
```
-c, --config arg        OCS2 MPC Config file
--verify            for MPC verification (default: 0)
```

### generate MPC offline file
```
-c, --config arg        OCS2 MPC Config file
--offline           for offline (default: 0)
--offline_file arg  Offline file
--offline_mode arg  Offline mode (0 or 1)
```

### start MPC server
```
-p, --port arg          Listening port
-c, --config arg        OCS2 MPC Config file
--server            for server (default: 0)
```

### run MPC simulation with Bullet
```
-c, --config arg        OCS2 MPC Config file
--urdf arg          URDF file
--simulation        for bullet simulation (default: 0)
--show_error        Show simulation error (default: 0)
```


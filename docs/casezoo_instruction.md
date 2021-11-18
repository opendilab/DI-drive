## DI-drive Casezoo Documentation

### Overview

DI-drive Casezoo consist of two parts: route scenario and single scenario. Route scenario can have individual scenarios
defined on the way. They are triggered when hero vehicle passed by and deleted after finishing the route.
Route scenario is defined in a *xml* file with its individual  scenarios and trigger location defined in a *json* file.
Single scenario is defined in a *python* file. If you want to run it individually, the [example](../core/data/example)
*xml* maps scenario in a certain location in a map, with each location a specific name. You can run it by passing its
name such as "ChangeLane_1".

### Usage
DI-drive provides a simple entry to runn Casezoo with *AutoPIDPolicy*. You can run [auto_run_case.py](../demo/auto_run/auto_run_case.py)
with following command:

``` bash
# Run route scenario
python auto_run_case.py --host [CARLA HOST] --port [CARLA_PORT] --route [ROUTE_FILE_PATH] [CONFIG_FILE_PATH]
# Run single scenario
python auto_run_case.py --host [CARLA HOST] --port [CARLA_PORT] --scenario [SCENARIO_NAME]
```

### Single Scenario

| Scenario name | Description | Sample image | File path |
| :---: | :--- | :---: | :---: |
| ControlLossNew | Hero vehicle briefly loses control and shakes | ![controlloss](figs/controlloss.png) | [control_loss_new.py](../core/simulators/srunner/scenarios/control_loss_new.py) <br> [ControlLoss.xml](../core/data/casezoo/example/ControlLoss.xml) |
| CutIn | There is a car behind the side quickly approaches and then cuts into the current lane of the vehicle, and then the speed is reduced to drive at a constant speed | ![cutin](figs/cutin.png) | [cut_in_new.py](../core/simulators/srunner/scenarios/cut_in_new.py) <br> [CutIn.xml](../core/data/casezoo/example/CutIn.xml) |
| FollowLeadingVehicleNew | Follow a slower vehicle ahead (turns, straights, ramps) | ![follow](figs/follow.png) | [follow_leading_vehicle_new.py](../core/simulators/srunner/scenarios/follow_leading_vehicle_new.py) <br> [LeadingVehicle.xml](../core/data/casezoo/example/LeadingVehicle.xml)|
| ChangeLane | There is a normal car near the front of the vehicle in the same lane, and a faulty car at a far distance. When the normal car approaches the faulty car, it changes lanes and cuts out (the picture shows a normal driving vehicle cut out)| ![changelane](figs/changelane.png)| [change_lane.py](../core/simulators/srunner/scenarios/change_lane.py) <br> [ChangeLane.xml](../core/data/casezoo/example/ChangeLane.xml)|
| OppositeDirection | There are vehicles in the opposite lane| ![opposite](figs/opposite.png)| [opposite_direction.py](../core/simulators/srunner/scenarios/opposite_direction.py) <br> [OppositeDirection.xml](../core/data/casezoo/example/OppositeDirection.xml)|
| SignalizedJunctionLeftTurn | The other car turns left at the traffic light (the other car can be in front of the car or on other roads at the intersection, and the car can go straight or turn)| ![left](figs/left.png)| [signalized_junction_left_turn.py](../core/simulators/srunner/scenarios/signalized_junction_left_turn.py) <br> [SignalizedJunctionLeftTurn.xml](../core/data/casezoo/example/SignalizedJunctionLeftTurn.xml)|
| SignalizedJunctionRightTurn | The other car turns right at the traffic light (the other car can be in front of the car or on other roads at the intersection, and the car can go straight or turn)| ![right](figs/right.png)| [signalized_junction_right_turn.py](../core/simulators/srunner/scenarios/signalized_junction_right_turn.py) <br> [SignalizedJunctionRightTurn.xml](../core/data/casezoo/example/SignalizedJunctionRightTurn.xml)|
| SignalizedJunctionStraight | The other car goes straight at the traffic light (the other car can be in front of the car or on other roads at the intersection, and the car can go straight or turn)| ![straight](figs/straight.png)| [signalized_junction_straight.py](../core/simulators/srunner/scenarios/signalized_junction_straight.py) <br> [SignalizedJunctionStraight.xml](../core/data/casezoo/example/SignalizedJunctionStraight.xml)|

### Route Scenario

| Route name | Sample image | Diffuculty | File path |
| :---: | :---: | :---: | :--- |
| route01 | ![route01](figs/route01.png) | Hard |[route01.xml](../core/data/casezoo/routes/route01.xml) <br> [route01.json](../core/data/casezoo/configs/route01.json)|
| route02 | ![route02](figs/route02.png)| Hard |[route02.xml](../core/data/casezoo/routes/route02.xml) <br> [route02.json](../core/data/casezoo/configs/route02.json)|
| route03 | ![route03](figs/route03.png)| Hard |[route03.xml](../core/data/casezoo/routes/route03.xml) <br> [route03.json](../core/data/casezoo/configs/route03.json)|
| route04 | ![route04](figs/route04.png)| Hard |[route04.xml](../core/data/casezoo/routes/route04.xml) <br> [route04.json](../core/data/casezoo/configs/route04.json)|
| route06 | ![route06](figs/route06.png)| Hard |[route06.xml](../core/data/casezoo/routes/route06.xml) <br> [route06.json](../core/data/casezoo/configs/route06.json)|
| route07 | ![route07](figs/route07.png)| Hard |[route07.xml](../core/data/casezoo/routes/route07.xml) <br> [route07.json](../core/data/casezoo/configs/route07.json)|
| route08 | ![route08](figs/route08.png)| Hard |[route08.xml](../core/data/casezoo/routes/route08.xml) <br> [route08.json](../core/data/casezoo/configs/route08.json)|
| route09 | ![route09](figs/route09.png)| Hard |[route09.xml](../core/data/casezoo/routes/route09.xml) <br> [route09.json](../core/data/casezoo/configs/route09.json)|
| town03_1 | ![town03_1](figs/town03_1.png)| Hard |[town03_1.xml](../core/data/casezoo/routes/town03_1.xml) <br> [town03_1.json](../core/data/casezoo/configs/town03_1.json)|
| town03_junctions01 | ![town03_junctions01](figs/town03_junctions01.png) | Easy | [town03_junctions01.xml](../core/data/casezoo/routes/town03_junctions01.xml) <br> [town03_junctions01.json](../core/data/casezoo/configs/town03_junctions01.json) |
| town03_junctions02 | ![town03_junctions02](figs/town03_junctions02.png)| Easy | [town03_junctions02.xml](../core/data/casezoo/routes/town03_junctions02.xml) <br> [town03_junctions02.json](../core/data/casezoo/configs/town03_junctions02.json) |
| town03_junctions03 | ![town03_junctions03](figs/town03_junctions03.png)| Easy | [town03_junctions03.xml](../core/data/casezoo/routes/town03_junctions03.xml) <br> [town03_junctions03.json](../core/data/casezoo/configs/town03_junctions03.json) |
| town04_junctions01 | ![town04_junctions01](figs/town04_junctions01.png)| Easy | [town04_junctions01.xml](../core/data/casezoo/routes/town04_junctions01.xml) <br> [town04_junctions01.json](../core/data/casezoo/configs/town04_junctions01.json) |
| town04_junctions02 | ![town04_junctions02](figs/town04_junctions02.png)| Easy | [town04_junctions02.xml](../core/data/casezoo/routes/town04_junctions02.xml) <br> [town04_junctions02.json](../core/data/casezoo/configs/town04_junctions02.json) |
| town04_junctions03 | ![town04_junctions03](figs/town04_junctions03.png)| Easy | [town04_junctions03.xml](../core/data/casezoo/routes/town04_junctions03.xml) <br> [town04_junctions03.json](../core/data/casezoo/configs/town04_junctions03.json) |
| town05_junctions01 | ![town05_junctions01](figs/town05_junctions01.png)| Easy | [town05_junctions01.xml](../core/data/casezoo/routes/town05_junctions01.xml) <br> [town05_junctions01.json](../core/data/casezoo/configs/town05_junctions01.json) |
| town05_junctions02 | ![town05_junctions02](figs/town05_junctions02.png)| Easy | [town05_junctions02.xml](../core/data/casezoo/routes/town05_junctions02.xml) <br> [town05_junctions02.json](../core/data/casezoo/configs/town05_junctions02.json) |
| town05_junctions03 | ![town05_junctions03](figs/town05_junctions03.png)| Easy | [town05_junctions03.xml](../core/data/casezoo/routes/town05_junctions03.xml) <br> [town05_junctions03.json](../core/data/casezoo/configs/town05_junctions03.json) |

You can debug a route by running its *xml* file with [no_scenarios.json](../core/data/casezoo/configs/no_scenarios.json). It runs no individual scenarios within the route.
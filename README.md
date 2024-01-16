# OptGBS
Optimal Solution Guided Branching Strategy for Neural Network Branch and Bound Verification

OptGBS is a branching strategy in the branch and bound neural network verification. 
This branching strategy is consisted of improvement estimation, out-of-bound compensation and score truncation. The improvements of the sub-problem produced by branching are estimated with the optimal solution of the parent problem.


## User Manmul
### Installation
First clone this repository via git as follows:
```bash
git clone https://github.com/xue-xy/OptGBS.git
cd OptGBS
```
Then install the python dependencies:
```bash
pip install -r requirements.txt
```
### Usage

#### ERAN Benchmark
```bash
python run.py --branch <branching strategy> --model <model name> --eps <radius> --tlimit <time> --batch_size <batch> --device <device>
```
+ `<branch>`: branching strategy, choice among 'optgbs', 'babsr', 'fsb', 'rand'.
+ `<model>`: the model you want to check.
+ `<eps>`: radius, float between 0 and 1.
+ `<tlimit>`: time limit for each property in seconds.
+ `<batch>`: batch size.
+ `<device>`: device to run the tool, cpu or cuda:0.


#### OVAL Benchmark
```bash
python oval_run.py --branch <branching strategy> --model <model name> --tlimit <time> --batch_size <batch> --device <device>
```
+ `<branch>`: branching strategy, choice among 'optgbs', 'babsr', 'fsb', 'rand'.
+ `<model>`: the model you want to check.
+ `<tlimit>`: time limit for each property in seconds.
+ `<batch>`: batch size.
+ `<device>`: device to run the tool, cpu or cuda:0.
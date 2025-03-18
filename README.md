# Elastic Body Module

This is a C++ static library that provides functionality for simulating Neo-Hookean body mechanics. The library is wrapped using `pybind11` to enable seamless integration with Python3.8.


## Class `ElasticBody`

### Constructor

```python

ElasticBody(E, lowerbound, upperbound, size, nv)

```

- **Parameters**:

- `E` (double): initalization of Young's modulus

- `lowerbound` (double): Lower bound of the computational domain

- `upperbound` (double): Upper bound of the computational domain

- `size` (int): sampling times

- `nv` (double): the Poisson ratio

### Functions

- `load_data`  :   Load sampling data (`body.load_data(cnt, filepath)`, where `cnt` is the starting count number of file: "pnt%i.txt"%cnt, "force%i.txt"%cnt, "A-%i.txt"%cnt, and "B-%i.txt"%cnt)

- `gen_grad_f`    : Generates internal force field (`body.gen_grad_f()`)      

- `gen_K`       :  Assemble initial stiffness matrix(`body.gen_K()`)            
        
- `eqn`         :   Compute forward simulation (`body.eqn(s)`, where `s` is the scale vector of Young's modulus in the bound) 

- `jac`        :    Compute Jacobian matrix  (`body.jac(s)`)

### Public Attributes

| Attribute        | Type           | Description                      |

| `we`             | `np.ndarray`        | Young's Modulus vector of cells        |

| `due_sum`            | `np.ndarray`   | Partial derivative of the surface deformation with respect to the Young's modulus for multiple presses    |

| `nodes_rt_list`  | `List[np.array]` | List of real-time nodes for sampling times      |

## Examples
We have provided example scripts and explanations to help using this library. 

`data` & `data-sampling`: The required input files in the problem;

`example.py`: An example script;

`draw.py`: An example script for visualization of the elastography; We have attached results, you can run this script directly by changing the path in the code to "we-ref.txt".
- The first column elastography is the ground truth, while the second is the reference result.

`results`: Results for the example models including indices and pictures. We also include the example results "we-ref.txt" with the corresponding indices and pictures in the "ref" subdirectiory for your convenience.


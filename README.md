# S-TLLR: STDP-inspired Temporal Local Learning Rule for Spiking Neural Networks

This source code implements the S-TLLR learning rule and replicates the experimental results obtained on the SHD, DVS Gesture, and DVS CIFAR-10 datasets. Its primary purpose is to aid in understanding the methodology and reproduce essential results.

## How to use:
Please, install the requirements listed in `requirements.txt`. Then, use the following command to run an experiment:

```shell
python main.py --arguments
```
Specific arguments are listed in the **Experiments** section to aid in reproducing some results from the paper. A description of each parameter is provided in `main.py`.

## S-TLLR implementation:
The S-TLLR implementation for linear, recurrent, and convolutional layers can be found in `./models/layers/STLLR_layers.py`. 

For instance, the S-TLLR implementation for a linear layer is divided into two classes: `LinearSTLLR` and `STLLRLinearGrad`. `LinearSTLLR` inherits from PyTorch's `nn.Linear` class, encapsulating all the variables required by the LIF models and detaching them from the PyTorch graph to remove temporal dependencies. The specific computation of the LIF model and S-TLLR occurs inside `STLLRLinearGrad`.

Here's a code snippet illustrating the key functionality:

```python
class STLLRLinearGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, trace_in, trace_out, mem, leak, threshold, factors):

        with torch.no_grad():
            # Trace of the pre-synaptic activity
            trace_in = factors[1] * trace_in + input
            
            # Leaky Integrate and Fire (LIF) computations
            output = F.linear(input, weight, bias)
            mem = torch.sigmoid(leak) * mem + output
            u_thr = mem - threshold.clamp(min=0.5)
            output = (u_thr > 0).float()
            
            # Trace of the post-synaptic activity 
            psi = 1 / torch.pow(100 * torch.abs(u_thr) + 1, 2) # secondary activation function
            trace_out_next = factors[0] * trace_out + psi
            
        ctx.save_for_backward(input, weight, bias, trace_in, trace_out, u_thr, threshold, factors)
        return output, mem, trace_in, trace_out_next

    @staticmethod
    def backward(ctx, grad_output, grad_mem):
        input, weight, bias, trace_in, trace_out, u_thr, threshold, factors = ctx.saved_tensors
        psi = 1/torch.pow(100*torch.abs(u_thr)+1, 2)
        
        # Learning signal propagation for next layer
        grad = psi*grad_output
        grad_input = torch.mm(grad, weight)
        
        # Elegibility traces computation [Equation (10)]
        delta_w_pre = factors[2]*trace_out.unsqueeze(2) * input.unsqueeze(1)
        delta_w_post = factors[3]*psi.unsqueeze(2) * trace_in.unsqueeze(1)
        
        # Weight updates [Equation (11)]
        grad_weight = (grad_output.unsqueeze(2)*(delta_w_post + delta_w_pre)).sum(0)
        grad_bias = None
        if bias is not None:
            grad_bias = grad.sum(dim=0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None
```

In the code, `trace_in` and `trace_out` represent traces of pre- and post-synaptic activities, computed forward-in-time. Weight updates with S-TLLR are computed by multiplying the learning signal (`grad_output`) with eligibility traces (`delta_w_pre + delta_w_post`).

This approach ensures that weight updates are temporally local, and memory complexity is proportional to the number of neurons, independent of the number of time-steps ($O(n)$).


## Experiments:
To replicate the results presented in Table 2, please follow these commands:
### DVS Gesture:
#### BPTT Baseline:
```shell
python main.py --dataset DVSGesture --arch dvs_vgg_bptt --save-path ./experiments/VGG_Gesture_BASELINE --data-path path_to_datasets_folder --trials 5 --epochs 300 --batch-size 16 --val-batch-size 16 --feedback-mode BP --print-freq 20 --delay-ls 5 --scheduler 300 --pooling MAX --training-mode bptt
```

#### S-TLLR:
```shell
python main.py --dataset DVSGesture --arch dvs_vgg_stllr --data-path path_to_datasets_folder --save-path ./experiments/VGG_Gesture_STLLR --trials 5 --epochs 300 --batch-size 16 --val-batch-size 64 --feedback-mode BP --print-freq 200 --delay-ls 5 --factors-stdp 0.2 0.75 -1 1 --pooling MAX --scheduler 300
```

### DVS CIFAR10:
#### BPTT Baseline:
```shell
python main.py --dataset CIFAR10DVS --arch dvscifar10_vgg_bptt --save-path ./experiments/VGG_CIFAR10DVS_BASELINE --data-path path_to_datasets_folder --trials 5 --epochs 300 --lr 0.001 --batch-size 48 --val-batch-size 128 --print-freq 20 --scheduler 300 --pooling AVG --activation GradSigmoid --training-mode bptt
```
#### S-TLLR:
```shell
python main.py --dataset CIFAR10DVS --arch dvscifar10_vgg11_stllr --save-path ./experiments/VGG11_CIFAR10DVS_STLLR --data-path path_to_datasets_folder --trials 5 --epochs 300 --lr 0.001 --batch-size 48 --val-batch-size 128 --feedback-mode BP --print-freq 20 --delay-ls 5 --scheduler 300 --factors-stdp 0.2 0.5 -1 1 --pooling AVG --activation STLLRConv2dSigmoid
```


### SHD:
#### BPTT Baseline:

```shell
python main.py --arch bptt_shd_net --dataset SHD --batch-size 128 --val-batch-size 128 --save-path ./experiments/SHD_BPTT_Baseline --print-freq 10 --data-path path_to_datasets_folder--trials 5 --epochs 200 --lr 0.0002 --training-mode bptt
```
#### S-TLLR:
- Using BP for the learning signal:
```shell
python main.py --arch stllr_shd_net --dataset SHD --batch-size 128 --val-batch-size 128 --factors-stdp 0.5 1 1 1 --delay-ls 90 --save-path ./experiments/SHD_STLLR_BP --print-freq 10 --data-path path_to_datasets_folder --trials 5 --epochs 200 --lr 0.0002
```

- Using DFA for the learning signal:
```shell
python main.py --arch stllr_shd_net --dataset SHD --batch-size 128 --val-batch-size 128 --factors-stdp 0.5 1 1 1 --delay-ls 90 --save-path ./experiments/SHD_STLLR_DFA --print-freq 10 --data-path path_to_datasets_folder --trials 5 --epochs 200 --lr 0.0002 --label-encoding one-hot --feedback-mode DFA
```

import torch
from torch.distributions.normal import Normal
from torch.nn import Linear
from torch.nn import ReLU
import numpy as np

def european_option_delta(log_moneyness, time_expiry, volatility) -> torch.Tensor:
    """
    Return Black-Scholes' delta of a European option.

    Parameters
    ----------
    - log_moneyness : torch.Tensor
        Log moneyness of the underlying asset.
    - time_expiry : torch.Tensor
        Time to the expiry of the European option.
    - volatility : torch.Tensor
        Volatility of the underlying asset.

    Returns
    -------
    delta : torch.Tensor

    Examples
    --------
    >>> european_option_delta([-0.01, 0.00, 0.01], 0.1, 0.2)
    tensor([0.4497, 0.5126, 0.5752])
    """
    s, t, v = map(torch.as_tensor, (log_moneyness, time_expiry, volatility))
    normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
    return normal.cdf((s + (v ** 2 / 2) * t) / (v * torch.sqrt(t)))


def clamp(x, lower, upper) -> torch.Tensor:
    """
    Clamp all elements in the input tensor into the range [`lower`, `upper`]
    and return a resulting tensor.
    The bounds `lower` and `upper` can be tensors.

    If lower <= upper:

        out = lower if x < lower
              x if lower <= x <= upper
              upper if x > upper

    If lower > upper:

        out = (lower + upper) / 2

    Parameters
    ----------
    - x : torch.Tensor, shape (*)
        The input tensor.
    - lower : float or torch.Tensor, default None
        Lower-bound of the range to be clamped to.
    - upper : float or torch.Tensor, default None
        Upper-bound of the range to be clamped to.

    Returns
    -------
    out : torch.tensor, shape (*)
        The output tensor.

    Examples
    --------
    >>> x = torch.linspace(-2, 12, 15) * 0.1
    >>> x
    tensor([-0.2000, -0.1000,  0.0000,  0.1000,  0.2000,  0.3000,  0.4000,  0.5000,
             0.6000,  0.7000,  0.8000,  0.9000,  1.0000,  1.1000,  1.2000])
    >>> clamp(x, torch.tensor(0.0), torch.tensor(1.0))
    tensor([0.0000, 0.0000, 0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000,
            0.7000, 0.8000, 0.9000, 1.0000, 1.0000, 1.0000])

    >>> x = torch.tensor([1.0, 0.0])
    >>> clamp(x, torch.tensor([0.0, 1.0]), torch.tensor(0.0))
    tensor([0.0000, 0.5000])
    """
    x = torch.min(torch.max(x, lower), upper)
    x = torch.where(lower < upper, x, (lower + upper) / 2)
    return x


class MultiLayerPerceptron(torch.nn.ModuleList):
    """
    Feed-forward neural network.

    Parameters
    ----------
    - in_features : int
        Number of input features.
    - out_features : int
        Number of output features.
    - n_layers : int, default 4
        Number of hidden layers.
    - n_units : int, default 32
        Number of units in each hidden layer.

    Examples
    --------
    >>> mlp = MultiLayerPerceptron(3, 1)
    >>> mlp
    MultiLayerPerceptron(
      (0): Linear(in_features=3, out_features=32, bias=True)
      (1): ReLU()
      (2): Linear(in_features=32, out_features=32, bias=True)
      (3): ReLU()
      (4): Linear(in_features=32, out_features=32, bias=True)
      (5): ReLU()
      (6): Linear(in_features=32, out_features=32, bias=True)
      (7): ReLU()
      (8): Linear(in_features=32, out_features=1, bias=True)
    )
    >>> mlp(torch.empty(2, 3))
    tensor([[...],
            [...]], grad_fn=<AddmmBackward>)
    """

    def __init__(self, in_features, out_features, n_layers=4, n_units=32):
        super().__init__()
        for n in range(n_layers):
            i = in_features if n == 0 else n_units
            self.append(Linear(i, n_units))
            self.append(ReLU())
        self.append(Linear(n_units, out_features))

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


def generate_geometric_brownian_motion(
    n_paths, maturity=30 / 365, dt=1 / 365, volatility=0.2, device=None
) -> torch.Tensor:
    """
    Return geometric Brownian motion of shape (n_steps, n_paths).

    Parameters
    ----------
    - n_paths : int
        Number of price paths.
    - maturity : float, default 30 / 365
        Length of simulated period.
    - dt : float, default 1 / 365
        Interval of time steps.
    - volatility : float, default 0.2
        Volatility of asset price.

    Returns
    -------
    geometric_brownian_motion : torch.Tensor, shape (N_STEPS, N_PATHS)
        Paths of geometric brownian motions. Here `N_PATH = int(maturity / dt)`.

    Examples
    --------
    >>> _ = torch.manual_seed(42)
    >>> generate_geometric_brownian_motion(3, maturity=5 / 365)
    tensor([[1.0000, 1.0000, 1.0000],
            [1.0023, 0.9882, 0.9980],
            [1.0257, 0.9816, 1.0027],
            [1.0285, 0.9870, 1.0112],
            [1.0405, 0.9697, 1.0007]])
    """
    randn = torch.randn((int(maturity / dt), n_paths), device=device)
    randn[0, :] = 0.0
    bm = volatility * (dt ** 0.5) * randn.cumsum(0)
    t = torch.linspace(0, maturity, int(maturity / dt))[:, None].to(bm)
    return torch.exp(bm - (volatility ** 2) * t / 2)


def entropic_loss(pnl) -> torch.Tensor:
    """
    Return entropic loss function, which is a negative of
    the expected exponential utility:

        loss(pnl) = -E[u(pnl)], u(x) = -exp(-x)

    Parameters
    ----------
    pnl : torch.Tensor, shape (*)
        Profit-loss distribution.

    Returns
    -------
    entropic_loss : torch.Tensor, shape (,)

    Examples
    --------
    >>> pnl = -torch.arange(4.0)
    >>> entropic_loss(pnl)
    tensor(7.7982)
    """
    return -torch.mean(-torch.exp(-pnl))


def to_premium(pnl) -> torch.Tensor:
    """
    Return the premium corresponding to the given profit-loss distribution.

    Premium is defined as the guaranteed amount of cash which is as preferable as
    the original profit-loss distribution in terms of the exponential utility.

    Parameters
    ----------
    pnl : torch.Tensor, shape (*)
        Profit-loss distribution.

    Returns
    -------
    premium : torch.Tensor, shape (,)

    Examples
    --------
    >>> pnl = -torch.arange(4.0)
    >>> premium = to_premium(pnl)
    >>> premium
    tensor(-2.0539)

    >>> torch.isclose(entropic_loss(torch.full_like(pnl, premium)), entropic_loss(pnl))
    tensor(True)
    """
    return -torch.log(entropic_loss(pnl))

def generate_heston_brownian_motion(n_paths=1000,maturity = 30 / 365, dt = 1/365, alpha = 0.2, sigma = 0.1,b= 0.05, V_0=0.04, corr = 0.2):
    # the Heston model written as dV = alpha (b- V(t)) dt + sigma sqrt(V(t)) dW_1(t)
    #                             dS/S = sqrt(V(t))dW_2(t)
    #                             corr(W1,W2)=corr
    #
    # The method we used is from the book Monte Carlo Methods in Finance, Glasserman 2003, Page 121 - 125
    # The default parameters lead to d = 4
    # The number of the trajectories usually is much larger than 1000, the default value is for test. 
    d = 4* b* alpha / sigma**2
    import math
    
    if d<=1:
        c = sigma**2 *(1-math.exp(-alpha*dt))/(4*alpha)
        lambda_const = math.exp(-alpha*dt)/c
        space1 = np.random.RandomState()
        randn_1 = space1.normal(size = (int(maturity/dt),n_paths))
        
        V = np.zeros(shape = (int(maturity/dt).n_paths))
        V[0,:]=V_0
        randn_1[0,:]=0

        for date_step in range(0,int(maturity/dt)-1):
            lambda_ = lambda_const * V[date_step]
            N = np.random.possion(lambda_/2,size=(1,n_paths))
            X = np.random.chisquare(df=N)
            V[date_step+1] = c*X
            
        brownian_process = randn_1*np.sqrt(V)*np.sqrt(dt)
        bp = brownian_process.cumsum(axis=0)
        bp = torch.from_numpy(bp)
        integral_t = (-0.5*V*dt)
        integral_t[0,:]=0
        integral_t=integral_t.cumsum(axis=0)
        integral_t = torch.from_numpy(integral_t)
        V = torch.from_numpy(V)
            
    else:
        c = sigma**2 *(1-math.exp(-alpha*dt))/(4*alpha)
        lambda_const = math.exp(-alpha*dt)/c
        space1 = np.random.RandomState()
        space2 = np.random.RandomState()
        space3 = np.random.RandomState()
        randn_1 = space1.normal(size = (int(maturity/dt),n_paths))
        randn_2 = space2.normal(size =(int(maturity/dt),n_paths)) * (1-corr**2)**0.5 + corr*randn_1
        chi_square = space3.chisquare(df=d-1,size=(int(maturity/dt),n_paths))
        V = np.zeros(shape = (int(maturity/dt),n_paths))
        V[0,:]=V_0
        randn_1[0,:]=0
        randn_2[0,:]=0
        for date_step in range(1,int(maturity/dt)):
            V[date_step] = c*((randn_2[date_step]+np.sqrt(lambda_const*V[date_step-1]))**2+chi_square[date_step-1])
        brownian_process = randn_1*np.sqrt(V)*np.sqrt(dt)
        bp = brownian_process.cumsum(axis=0)
        bp = torch.from_numpy(bp)
        integral_t = (-0.5*V*dt)
        integral_t[0,:]=0
        integral_t=integral_t.cumsum(axis=0)
        integral_t = torch.from_numpy(integral_t)
        V = torch.from_numpy(V)

    
    dS2_1 = V*dt
    S2_1 = np.cumsum(dS2_1,axis=0)
    t = np.full(shape=(int(maturity/dt),n_paths),fill_value=dt)
    t = np.cumsum(t,axis=0)
    t = torch.from_numpy(t)
    L_t_v = (V-b)/alpha*(1-np.exp(-alpha*(maturity-t)))+b*(maturity-t)
    S2 = S2_1+L_t_v
        
    return (torch.exp(integral_t+bp).to(torch.float32),S2.to(torch.float32))

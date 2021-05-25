import torch

# https://arxiv.org/pdf/2012.11230.pdf

class RoundDiff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class DAQBase:
    def __init__(self, n_bits=8):
        self.step_sizes = {1: 1.596, 2: 0.996, 3: 0.586, 4: 0.335, 8: 0.031}
        self.n_bits = n_bits
        self.step_size = self.step_sizes[n_bits]
        self.alpha = (2 ** (self.n_bits - 1) - 0.5) * self.step_size
        self.beta = 0

        self.mean = 0
        self.sigma = 0

    def quantize(self, x: torch.Tensor):
        pass


class DAQActivations(DAQBase):
    def __init__(self, n_bits=8, daq_input=False):
        super().__init__(n_bits=n_bits)
        self.daq_input = daq_input

    def adaptive_transformer(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        self.sigma = torch.std(x, dim=(0, 2, 3), keepdim=True)
        return (x - self.mean) / (self.sigma + 1e-7)

    def inv_adaptive_transformer(self, x_q_hat: torch.Tensor) -> torch.Tensor:
        return x_q_hat * self.sigma + self.mean

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        self.update_beta()

        x_clamped = torch.zeros_like(x)
        for c in range(x.shape[1]):
            x_clamped[:, c, ...] = torch.clamp(x[:, c, ...], -self.alpha + self.beta[0, c, 0, 0].item(),
                                               self.alpha + self.beta[0, c, 0, 0].item())
        return x_clamped

    def update_beta(self):
        if self.daq_input:
            self.beta = torch.zeros_like(self.sigma)
        else:
            self.beta = torch.maximum(self.alpha - self.mean / (self.sigma + 1e-7), torch.zeros_like(self.sigma))

    def adaptive_discretizer(self, x_hat: torch.Tensor) -> torch.Tensor:
        return (RoundDiff.apply((self.clamp(x_hat) / self.step_size) + 0.5) - 0.5) * self.step_size

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.adaptive_transformer(x)
        x_q_hat = self.adaptive_discretizer(x_hat)
        x_q = self.inv_adaptive_transformer(x_q_hat)
        return x_q


class DAQWeights(DAQBase):
    def __init__(self, n_bits=8):
        super().__init__(n_bits=n_bits)

    def adaptive_transformer(self, x: torch.Tensor) -> torch.Tensor:
        # weight quantizer is not adaptive to channels
        self.sigma = torch.std(x, dim=0, keepdim=True)
        return x / (self.sigma + 1e-7)

    def inv_adaptive_transformer(self, x_q_hat: torch.Tensor) -> torch.Tensor:
        return x_q_hat * self.sigma

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -self.alpha, self.alpha)
        return x

    def adaptive_discretizer(self, x_hat: torch.Tensor) -> torch.Tensor:
        return (RoundDiff.apply((self.clamp(x_hat) / self.step_size) + 0.5) - 0.5) * self.step_size

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.adaptive_transformer(x)
        x_q_hat = self.adaptive_discretizer(x_hat)
        x_q = self.inv_adaptive_transformer(x_q_hat)
        return x_q


def main():
    daq_w = DAQWeights(n_bits=8)
    w = torch.rand(32, 16, 3, 3)
    w_q = daq_w.quantize(w)
    print(f'w: {w.shape} w_q: {w_q.shape}')

    daq_act = DAQActivations(n_bits=4, daq_input=False)
    x = torch.rand(1, 3, 64, 64)
    x_q = daq_act.quantize(x)
    print(f'x: {x.shape} x_q: {x_q.shape}')

    daq_input = DAQActivations(n_bits=4, daq_input=True)
    x = torch.rand(1, 3, 64, 64)
    x_q = daq_input.quantize(x)
    print(f'x: {x.shape} x_q: {x_q.shape}')


if __name__ == "__main__":
    main()

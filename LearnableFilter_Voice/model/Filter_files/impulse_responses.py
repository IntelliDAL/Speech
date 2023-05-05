import math
import scipy
from torch import nn, sinc, special
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
from math import ceil
import torch
from torch.distributions import Normal


def gabor_impulse_response(t, center, fwhm):
    denominator = 1. / (torch.sqrt(torch.tensor(2.0) * math.pi) * fwhm)  # 分母
    gaussian = torch.exp(-(torch.tensordot(1.0 / (2. * fwhm.unsqueeze(1) ** 2), (t ** 2.).unsqueeze(0), dims=1)))
    center_frequency_complex = center.type(torch.complex64)
    t_complex = t.type(torch.complex64)
    sinusoid = torch.exp(
        torch.complex(torch.tensor(0.), torch.tensor(1.)) * torch.tensordot(center_frequency_complex.unsqueeze(1),
                                                                            t_complex.unsqueeze(0), dims=1))
    denominator = denominator.type(torch.complex64).unsqueeze(1)
    gaussian = gaussian.type(torch.complex64)

    return denominator * sinusoid * gaussian


def gabor_impulse_response_legacy_complex(t, center, fwhm):
    denominator = 1. / (torch.sqrt(torch.tensor(2.0) * math.pi) * fwhm)  # 分母
    gaussian = torch.exp(-(torch.tensordot(1.0 / (2. * fwhm.unsqueeze(1) ** 2), (t ** 2.).unsqueeze(0), dims=1)))
    temp = torch.tensordot(center.unsqueeze(1), t.unsqueeze(0), dims=1)
    temp2 = torch.zeros(*temp.shape + (2,), device=temp.device)

    # since output of torch.tensordot(..) is multiplied by 0+j
    # output can simply be written as flipping real component of torch.tensordot(..) to the imag component

    temp2[:, :, 0] *= -1 * temp2[:, :, 0]
    temp2[:, :, 1] = temp[:, :]

    # exponent of complex number c is
    # o.real = exp(c.real) * cos(c.imag)
    # o.imag = exp(c.real) * sin(c.imag)

    sinusoid = torch.zeros_like(temp2, device=temp.device)
    sinusoid[:, :, 0] = torch.exp(temp2[:, :, 0]) * torch.cos(temp2[:, :, 1])
    sinusoid[:, :, 1] = torch.exp(temp2[:, :, 0]) * torch.sin(temp2[:, :, 1])

    # multiplication of two complex numbers c1 and c2 -> out:
    # out.real = c1.real * c2.real - c1.imag * c2.imag
    # out.imag = c1.real * c2.imag + c1.imag * c2.real

    denominator_sinusoid = torch.zeros(*temp.shape + (2,), device=temp.device)
    denominator_sinusoid[:, :, 0] = (
            (denominator.view(-1, 1) * sinusoid[:, :, 0])
            - (torch.zeros_like(denominator).view(-1, 1) * sinusoid[:, :, 1])
    )
    denominator_sinusoid[:, :, 1] = ((denominator.view(-1, 1) * sinusoid[:, :, 1])
                                     + (torch.zeros_like(denominator).view(-1, 1) * sinusoid[:, :, 0])
                                     )

    output = torch.zeros(*temp.shape + (2,), device=temp.device)

    output[:, :, 0] = ((denominator_sinusoid[:, :, 0] * gaussian)
                       - (denominator_sinusoid[:, :, 1] * torch.zeros_like(gaussian))
                       )
    output[:, :, 1] = ((denominator_sinusoid[:, :, 0] * torch.zeros_like(gaussian))
                       + (denominator_sinusoid[:, :, 1] * gaussian)
                       )
    return output


def gabor_filters(kernel, size: int = 401, legacy_complex=False):
    t = torch.arange(-(size // 2), (size + 1) // 2, dtype=kernel.dtype, device=kernel.device)
    if not legacy_complex:
        impulse_response = gabor_impulse_response(t, center=kernel[:, 0], fwhm=kernel[:, 1])
        return impulse_response
    else:
        return gabor_impulse_response_legacy_complex(t, center=kernel[:, 0], fwhm=kernel[:, 1])


def gaussian_lowpass(sigma, filter_size: int):
    # sigma = torch.clamp(sigma, min=(2. / filter_size), max=0.5)
    t = torch.arange(0, filter_size, dtype=sigma.dtype, device=sigma.device)
    t = torch.reshape(t, (1, filter_size, 1, 1))
    numerator = t - 0.5 * (filter_size - 1)
    denominator = sigma * 0.5 * (filter_size - 1)
    return torch.exp(-0.5 * (numerator / denominator) ** 2)


def window(alpha, N):
    def hamming(n, i):
        ham = (1 - i) - i * torch.cos(torch.tensor(2) * math.pi * (n - 1) / (N - 1)).cuda()
        return ham

    kernels = torch.zeros((alpha.shape[0], N))
    for i, data in enumerate(alpha):
        kernel = torch.asarray([hamming(n, data) for n in range(N)])
        kernels[i] = kernel
    return kernels.unsqueeze(1)


def Sincwindows(N_filt, Filt_dim, fs, filt_b1):
    freq_scale = fs * 1.0
    # window_ = torch.hamming_window(Filt_dim)
    # window_ = torch.kaiser_window(Filt_dim)
    # window_ = torch.blackman_window(Filt_dim)
    filters = Variable(torch.zeros((N_filt, Filt_dim))).cuda()
    N = Filt_dim
    t_right = Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / fs).cuda()
    min_freq = 50.0
    filt_beg_freq = torch.abs(filt_b1) + min_freq / freq_scale
    for i in range(N_filt):
        low_pass1 = 2 * filt_beg_freq[i].float() * Sinc(filt_beg_freq[i].float() * freq_scale, t_right)
        low_pass1 = low_pass1 / torch.max(low_pass1)
        filters[i, :] = low_pass1.cuda()
        # filters[i, :] = low_pass1 * window_.cuda()

    return filters.view(N_filt, 1, Filt_dim)


def Sinc(band, t_right):
    y_right = torch.sinc(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = flip(y_right, 0)
    y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])
    return y


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
        getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def bilateralFtr1D(y, sSpatial, sIntensity):
    '''
    The equation of the bilateral filter is

            (       dx ^ 2       )       (         dI ^2        )
    F = exp (- ----------------- ) * exp (- ------------------- )
            (  sigma_spatial ^ 2 )       (  sigma_Intensity ^ 2 )
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        This is a guassian filter!
    dx - The 'geometric' distance between the 'center pixel' and the pixel to sample
    dI - The difference between the intensity of the 'center pixel' and the pixel to sample
    sigma_spatial and sigma_Intesity are constants. Higher values mean
    that we 'tolerate more' higher value of the distances dx and dI.
    Dependencies: numpy, scipy.ndimage.gaussian_filter1d

    calc gaussian kernel size as: filterSize = (2 * radius) + 1; radius = floor (2 * sigma_spatial)
    y - input data
    '''

    # gaussian filter and parameters
    radius = torch.floor(2 * sSpatial)
    filterSize = ((2 * radius) + 1)
    kernel = filterSize.shape
    ftrArray = torch.zeros(kernel).cuda()
    ftrArray[radius] = 1
    # Compute the Gaussian filter part of the Bilateral filter
    gauss = gaussian_filter1d(ftrArray, sSpatial)
    # 1d data dimensions
    width = y.size
    # 1d resulting data
    ret = torch.zeros(width)

    for i in range(width):
        ## To prevent accessing values outside of the array
        # The left part of the lookup area, clamped to the boundary
        xmin = max(i - radius, 1)
        # How many columns were outside the image, on the left?
        dxmin = xmin - (i - radius)
        # The right part of the lookup area, clamped to the boundary
        xmax = min(i + radius, width)
        # How many columns were outside the image, on the right?
        dxmax = (i + radius) - xmax
        # The actual range of the array we will look at
        # The left expression in the bilateral filter equation
        # We take only the relevant parts of the matrix of the
        # Gaussian weights - we use dxmin, dxmax, dymin, dymax to
        # ignore the parts that are outside the image
        expS = gauss[(1 + dxmin):(filterSize - dxmax)]

        # The right expression in the bilateral filter equation
        dy = y[xmin:xmax] - y[i]
        dIsquare = (dy * dy)
        expI = torch.exp(- dIsquare / (sIntensity * sIntensity))

        # The bilater filter (weights matrix)
        F = expI * expS

        # Normalized bilateral filter
        Fnormalized = F / sum(F)

        # Multiply the area by the filter
        tempY = y[xmin:xmax] * Fnormalized

        # The resulting pixel is the sum of all the pixels in
        # the area, according to the weights of the filter
        # ret(i,j,R) = sum (tempR(:))
        ret[i] = sum(tempY)

    return ret


def gaussian_kernel_1d(sigma: float, num_sigmas: float = 3.) -> torch.Tensor:
    radius = ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())


def Kaiser_windows(M, beta):
    beta = M * beta
    beta = torch.clamp(beta, min=0.01, max=50)
    n = torch.arange(0, M, device=beta.device)
    alpha = (M - 1) / 2.0
    w = (special.i0(beta * torch.sqrt(1 - ((n - alpha) / alpha) ** 2.0).cuda()) / special.i0(beta))
    return w


def boxcarWindow_(N_filt, Filt_dim, fs, filt_b1):
    t = torch.arange(1, Filt_dim+1, dtype=filt_b1.dtype, device=filt_b1.device) / 400
    t = torch.reshape(t, (1, Filt_dim, 1, 1))
    low_pass1 = nn.functional.relu(filt_b1 - torch.abs(t)) / (torch.abs(filt_b1 - torch.abs(t)) + torch.tensor(1e-10, device=t.device)) / Filt_dim
    print(filt_b1[0])
    return low_pass1


def boxcarWindow(N_filt, Filt_dim, fs, filt_b1):
    filters = Variable(torch.zeros((N_filt, Filt_dim))).cuda()
    t_right = Variable(torch.ones(int((Filt_dim - 1) / 2)), requires_grad=True).cuda()
    filt_beg_freq = Variable(filt_b1, requires_grad=True)
    for i in range(N_filt):
        low_pass1 = rectangle_(filt_beg_freq[i], t_right, fs)
        filters[i, :] = low_pass1.cuda()
    return filters.view(N_filt, 1, Filt_dim)


def rectangle_(band, y_right, fs):
    index = Variable((torch.round(band * fs)).to(torch.float64), requires_grad=True)
    # y_right[y_right > index] = 0.0
    for i in range(len(y_right)):
        if i > index:
            y_right[i] = 0.0
    print(band)
    y_left = flip_(y_right.cuda()).squeeze()
    y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])
    return y


def flip_(x):
    x = x.view(-1, 1)
    x = torch.flip(x, dims=(1, 0))
    return x

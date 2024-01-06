# -------- Backwards Compatibility ---------
from __future__ import print_function
from __future__ import division
from builtins   import zip
from builtins   import range
from builtins   import object
# -------------- Utilities -----------------
from scipy.special import loggamma
import copy
import numpy as np
# ----------- Relative Imports -------------
from ..config import V0_TILDE, ALPHA, HBARC, HBAR_MEC, MEC2, KAPPA
from ..config import DMASS_NH, R0, MN, EPSILON, GA, GV, LAM
from ..config import DEFAULTS
# ------------------------------------------

__version__ = u'2.0.0'
__date__    = u'2019-07-26'

#=========================================================================#
#                           Class phaseSpace                              #
#=========================================================================#
class phaseSpace(object):
    """
    An instance of the class phaseSpace contains the methods for computing
    phase space integrals and fits to those integrals which can be analytically
    continued to the complex plane.

    This is designed to work similarly to the fortProcess objects, in that
    it has default settings that are overridden.

    Args:
        beta (str): The beta decay type (ONLY '-' IS IMPLEMENTED)

    Attributes:
        beta (str): The beta decay type.
        _settings (dict): The underlying phase space settings.
    """

    def __init__(self, beta):
        # type of beta decay ('+', '-', 'ec')
        self.beta = beta
        if beta not in [u'-']:
            raise ValueError(u"The phase space for the requested beta decay type is not implemented.")

        # default settings
        self._settings = self._getDefaults()

    @property
    def approx(self):
        """ approx (str): The type of approximation used to analytically continue
        the phase space integrals.
        """
        return self._settings[u'psi_approx']

    #-----------------------------------------------------------------------
    def updateSettings(self, override):
        """
        Update the settings for the phase space calculations.

        Since psi_approx is itself available as override setting, and each value
        of psi_approx has different settings (like the famContour) we have to
        update the available settings if we change it.

        Args:
            override (dict): Settings to override.
        """
        if u'psi_approx' in list(override.keys()):
            self._settings = self._getDefaults(override[u'psi_approx'])

        for h in override:
            if h not in list(self._settings.keys()):
                raise KeyError(u"Invalid override setting '{:}' ".format(h) +\
                        u"for phase space approx '{:}'.".format(self.approx))
            else:
                self._settings[h] = override[h]

    #-----------------------------------------------------------------------
    def psiFct(self, n, Zd, A, eqrpamax, eqrpamin=0, debug=False):
        """
        Construct a function representing an analytic approximation of the
        phase space integrals along the real axis.

        Two types of approximations are implemented:

            * RATINT: A rational function interpolation of the integrals on the real axis.
            * POLYFIT: A polynomial least squares fit of the integrals on the real axis.

        The function is constructed as follows:

            1) Contruct a grid of maximum electron energies :math:`W_0=1+(EQRPA_{max} - EQRPA)/m_ec^2`
                along the real axis at chebychev nodes (with ratint_pts, or polynomial_fit_pts).
            2) Calculate the phase space integral f_n(w0) with  gauss-legendre quadrature using
                psi_gltpts points.
            3) Feed :math:`x=w_{0_{cheby}}, y=f_n(w_0)` into polynomial fit or rational fct interpolation.
            4) Check accuracy of fit to fit points.

        Args:
            n (int): The index for the phase space type (Mustonent et al., Phys. Rev. C 90, 024308 (2014)).
            Zd (int): Charge of the daughter nucleus.
            A (int): Mass of the daughter/parent nucleus.
            eqrpamax (float): Maximum QRPA energy for beta decay.
            eqrpamin (float): Minimum QRPA energy for phase space integral approximation (default 0)
            debug (bool): If true, also returns the x=w0, y=psi values used for the approximation.

        Returns:
            ufunc
            ndarray: The x=w0 values used for the approximation (only if debug=True).
            ndarray: The y=psi values used for the approximation (only if debug=True).
        """
        # Electron energy: Min = mec^2, Max = mec^2 + EQRPA_max
        #                      => 1         => 1 + EQRPA_max/MEC2
        w0min = 1.0
        w0max = 1.0 + ((eqrpamax-eqrpamin)/MEC2)

        # Get chebychev grid of 'x-values' (w0) along real axis for the approximation
        if self.approx == u'RATINT':
            npts_approx = self._settings[u'ratint_pts']
        elif self.approx == u'POLYFIT':
            npts_approx = self._settings[u'polynomial_fit_pts']

        w0_vals, cwts = np.polynomial.chebyshev.chebgauss(npts_approx)
        w0_vals, cwts = transform_interval(w0_vals,cwts,w0min,w0max)

        # Calculate the corresponding 'y-values' (phase space integrals)
        psi = np.array([self.calcPsi(n, Zd, A, w0, debug=False) for w0 in w0_vals])

        # Contruct the approximation using known x=w0grid and y=f_n(w0)
        if self.approx == u'RATINT':
            psi_approx = self.thieleInterpolator(w0_vals, psi)
        elif self.approx == u'POLYFIT':
            order = self._settings[u'polynomial_order']
            psi_approx = self.polynomialFit(w0_vals, psi, order)

        # Define a wrapper fct so if we're on the real axis we don't use the approx
        def psi_fct(xin):
            # Convert to iterable (in case xin is a point, numpy array, list, etc)
            try:
                iter(xin)
            except TypeError:
                xin = np.array([xin])

            # Check if on the real axis and return either psi or approximation
            if np.isreal(xin).all():
                #psi_out = np.zeros(len(xin), dtype=np.float_)
                #for i, xx in enumerate(xin):
                #    psi_out[i] = self.calcPsi(n, Zd, A, xx, debug=False)

                # This is too slow. Until we implement arrays over for-loops
                # use the approx unless the result should be zero, then force zero.
                psi_out = psi_approx(xin)
                # EQRPA > EQRPAmax is meaningless for phase space, force to zero.
                inds = np.where(xin <= 1.0)[0]
                if len(inds) != 0: psi_out[inds] = 0.0
            else:
                psi_out = psi_approx(xin)
            return psi_out

        # Return the wrapper fct (+ grid used for approx if debug on)
        if not debug:
            return psi_fct
        else:
            return psi_fct, w0_vals, np.array(psi)

    #-----------------------------------------------------------------------
    def calcPsi(self, n, Zd, A, w0, debug=False):
        """
        Calculate the phase space integral from 1 to w0 with Gauss-
        Legendre quadrature on psi_glpts sized grid (from settings).

        Args:
            n (int): The index for the phase space type (Mustonent et al., Phys. Rev. C 90, 024308 (2014)).
            Zd (int): Charge of the daughter nucleus.
            A (int): Mass of the daughter/parent nucleus.
            w0 (float): Maximum electron energy for the decay :math:`W_0=1+(EQRPA_{max} - EQRPA)/m_ec^2`
            debug (bool): If True, also return the x=w and y=phase space values used for the integration.

        Returns:
            float
            ndarray: The x=w values used for the integration (only if debug=True).
            ndarray: The y=phase space used for the integration (only if debug=True).
        """

        if n not in list(range(1,7)):
            raise ValueError(u"Phase space factor n takes values 1 to 6.")

        if w0 <= 1.0:
            #print(u"WARNING: Maximum electron energy cannot be less than rest mass.")
            #print(u"         Setting phase space integral to zero...")
            return 0.0

        if debug: farr = []
        fint = 0.0
        wmin = 1.0
        glpts = self._settings[u'psi_glpts']
        w, glwts = np.polynomial.legendre.leggauss(glpts)
        w, glwts = transform_interval(w, glwts, wmin, w0)

        for wx, gl in zip(w,glwts):
            if n == 1:
                gn = gam_ke(1,Zd)
            elif n==2:
                gn = wx
            elif n==3:
                gn = wx**2
            elif n==4:
                gn = wx**3
            elif n==5:
                gn = wx*(w0-wx)**2
            elif n==6:
                gn = wx*p2lambda2(Zd,A,wx)
            px = np.sqrt(wx**2.0 - 1.0)
            f = px*(w0-wx)**2.0*L0(Zd)*Fermi(0,Zd,A,wx)*gn

            if not np.isfinite(f):
                print(u"WARNING: phase space integrand is infinite (w ~ 1.0).")
                print(u"         Setting value to zero...")
                f = 0.0

            fint += f*gl
            if debug: farr.append(f)

        if not debug:
            return fint
        else:
            return fint, w, farr

    #-----------------------------------------------------------------------
    def polynomialFit(self, x, y, d):
        """
        Returns a function which evaluates a polynomial least squares fit
        built from x and y.

        Args:
            x (ndarray): The x values for the fit.
            y (ndarray): The y values for the fit.
            d (int): The degree of the polynomial used for the fit.

        Returns:
            ufunc
        """
        def p(xin):
            return np.polyval(np.polyfit(x,y,d), xin)
        return p

    #-----------------------------------------------------------------------
    def thieleInterpolator(self, x, y):
        """
        Returns a funcion which evaluates a rational function interpolation
        built from x and y using thiele continued fractions.

        Args:
            x (ndarray): The x values for the fit.
            y (ndarray): The y values for the fit.

        Returns:
            ufunc
        """
        # Calculate the continued fraction coefficients
        p = [[yi]*(len(y)-i) for i, yi in enumerate(y)]
        for i in range(len(p)-1):
            p[i][1] = (x[i] - x[i+1])/(p[i][0] - p[i+1][0])
        for i in range(2, len(p)):
            for j in range(len(p)-i):
                p[j][i] = (x[j]-x[j+i])/(p[j][i-1]-p[j+1][i-1]) + p[j+1][i-2]
        p0 = p[0]

        # Evaluate the continued fraction
        def t(xin):
            a = 0
            for i in range(len(p0)-1, 1, -1):
                a = ((xin - x[i-1])/(p0[i] - p0[i-2] + a))
            return y[0] + ((xin-x[0])/(p0[1]+a))

        # Check the interpolation passes though the provided points
        for xx, yy in zip(x, y):
            diff   = abs(yy - t(xx))
            mean   = 0.5*(abs(yy + t(xx)))
            reldiff = (diff/(mean+EPSILON))
            if min(reldiff,diff) > 1e-5:
                print(u"WARNING: ratint does not pass through points precisely.")
                print(u"Relative diff: {:}, Abs diff: {:} at x={:}.".format(reldiff, diff, xx))

        # Return the evaluating function
        return t

    #-----------------------------------------------------------------------
    def _getDefaults(self, psi_approx=None):
        """
        Return the dictionary for default phase space settings.

        Args:
            psi_approx (str): The default approximation type (default None --> config.py value).

        Returns:
            dict
        """

        default_raw = copy.deepcopy(DEFAULTS[u'psi'])

        default_settings = default_raw[u'PSI']
        if psi_approx is None:
            psi_approx = default_settings[u'psi_approx']
        default_settings.update(default_raw[psi_approx])
        default_settings[u'psi_approx'] = psi_approx # this is redundant, but just to be safe

        return default_settings

#=========================================================================#
#                   Fermi Function Related Fcts                           #
#=========================================================================#
def transform_interval(x, wts, xmin, xmax):
    """
    Given gauss legendre grid and weights on -1 to 1, transform
    to interval xmin to xmax.

    Args:
        x (ndarray): The Gauss-Legendre grid on -1 to 1.
        wts (ndarray): The Gauss-Legendre weights on -1 to 1 grid.
        xmin (float): New lower bound.
        xmax (float): New upper bound.

    Returns:
        ndarray: Transformed Gauss-Legendre grid.
        ndarray: Transformed Gauss-Legendre weights.
    """
    f1  = 0.5*(xmin+xmax)
    f2  = 0.5*(xmax-xmin)
    wts = wts*f2
    x   = x*f2 + f1
    return x, wts

#-----------------------------------------------------------------------
def V0_shift(Zd):
    """
    Positive energy shift due to charge screening from a daughter nucleus
    of charge Zd.

    Args:
        Zd (int): Charge of daughter nucleus.

    Returns:
        float
    """
    if Zd < 0: raise ValueError(u"Zd < 0 not supported in V0_shift.")
    return V0_TILDE*ALPHA**2*(Zd-1)**(4.0/3.0)

#-----------------------------------------------------------------------
def w_screen(Zd, w):
    r"""
    Shifted electron energy due to charge screening (denoted :math:`\tilde{w}`).

    Args:
        Zd (int): Charge of daughter nucleus.
        w (float): Electron energy.

    Returns:
        float
    """
    return w - V0_shift(Zd)

#-----------------------------------------------------------------------
def gam_ke(ke, Zd):
    """
    Appears in the definition of the Fermi Functions. For first forbidden
    we have gamma1 or gamma2 generalized to gamma_ke.

    Args:
        ke (int): .
        Zd (int): Charge of daughter nucleus.

    Returns:
        float
    """
    return np.sqrt(float(ke)**2.0 - (ALPHA*Zd)**2.0)

#-----------------------------------------------------------------------
def L0(Zd, sc=False):
    """
    Simplest approximation for L0 without charge-screening.

    Args:
        Zd (int): Charge of daughter nucleus.
        sc (bool): Include corrections for charge screening.

    Returns:
        float
    """
    if not sc:
        return 0.5*(1.0 + gam_ke(1, Zd))
    else:
        raise ValueError(u"Charge screening not yet supported for L0.")

#-----------------------------------------------------------------------
def L0_2(Zd, A, w):
    r"""
    Approximation to L0 for :math:`|Z| <~ 15`.

    Args:
        Zd (int): Charge of daughter nucleus.
        A (int): Maxx of parent/daughter nucleus.
        w (float): Electron energy.

    Returns:
        float
    """
    R = ((R0/HBAR_MEC))*A**(1.0/3.0)
    L0_2 = 1.0 - ALPHA*Zd*w*R + (13.0/60.0)*(ALPHA*Zd)**2 - 0.5*((ALPHA*Zd*R/w))
    return L0_2

#-----------------------------------------------------------------------
def Fermi(F, Zd, A, w, sc=False):
    r"""
    Fermi Function, excluding L0 which must be multiplied separately.

    We first compute ln(F) to avoid overflow and underflow in individual
    terms. Once combined the value is reasonable to then exponentiate.

    Note:
        scipy loggamma handles complex arguments just fine, see scipy documentation
        for difference between loggamma and gammaln.

        About numerical stability:
        As :math:`w \rightarrow 1`, y become large (~10^4), thus :math:`\exp(\pi*y)` overflows.
        For large y, the Re and Im parts of :math:`\gamma(1+iy)` underflow, approaching zero.
        Taking the :math:`ln`, :math:`\ln \exp(\pi*y) \propto y` and
        :math:`\ln \gamma(1+iy) \propto -y` (not sure exact relation).
        These roughly balance when added, allowing us to exponentiate the result and
        recover the Fermi function avoiding over/underflows.

    Args:
        F (int): The type of Fermi function 0 or 1.
        Zd (int): Charge of the daughter nucleus.
        A (int): Mass of the parent/daughter nucleus.
        w (float): The electron energy.
        sc (bool): Include correction for charge screening (default False).

    Returns:
        float

    Raises:
        ValueError
    """

    if F not in [0,1]:
        raise ValueError(u"F takes value 0 for Fermi function F0 or 1 for"+\
                u" generalized Fermi function F1.")
    # This is not programmed for positron decay
    if Zd < 0: raise ValueError(u"Zd < 0 not allowed in Fermi.")
    # Warn if Re(w) < 1
    if np.real(w) < 1.0: print(u"WARNING: Re(w) < 1 in Fermi.")

    # Charge screening
    if not sc:
        pref = 1.0
        p = np.sqrt(w**2.0 - 1.0)
    else:
        # w becomes w tilde
        w_us = w
        w    = w_screen(Zd, w)
        # Energy conservation
        if np.real(w) < 1.0: return 0.0
        p_us = np.sqrt(w_us**2.0 - 1.0)
        p    = np.sqrt(w**2.0 - 1.0)
        pref = ((p/p_us))*((w/w_us))

    # Special case: when p --> 0, shape should go to 0, so we can safely return 0
    if abs(w-1.0) < EPSILON: return 0.0

    R = (R0/HBAR_MEC*A**(1.0/3.0))
    y = (ALPHA*Zd*w/p)
    # ([ke* (2ke-1)!!]^2) * (4^ke) * (2pR)^(2gam_ke - ke)
    if F == 0:
        gF = gam_ke(1, Zd)
        factor = 4.0*(2.0*p*R)**(2.0*gF-2.0)
    elif F == 1:
        gF = gam_ke(2, Zd)
        factor = 576.0*(2.0*p*R)**(2.0*gF-4.0)


    ln_expy = np.pi*y
    garg = gF + 1j*y
    ln_gamnum1 = loggamma(garg)
    ln_gamnum2 = np.conj(loggamma(garg))
    ln_gamden  = loggamma(2.0*gF + 1.0)

    ln_F= np.log(pref) +np.log(factor) +ln_expy +ln_gamnum1 +ln_gamnum2 -2*ln_gamden

    # F contains abs(gamma(x+iy))^2 so should be real, thus ln F should be real
    if np.imag(ln_F) > EPSILON:
        print(u"WARNING: ln(F(w)) is not exactly real, but it should be.")
    # This should never happen because 'garg' is complex and 'gF' is a sqrt and thus > 0,
    # but gamma/loggamma is undefined on the negative real axis...
    if any(np.isnan(x) for x in [ln_gamnum1, ln_gamden]):
        print(u"WARNING: gamma function in phase space calculation received arg "+\
               "on the negative real axis. This is undefined.")

    # Note: abs(lnF) where im(F) = 0 should give us just the real part
    return np.exp(abs(ln_F))

#-----------------------------------------------------------------------
def p2lambda2(Zd, A, w, sc=False):
    r"""
    First approximation to the Coulomb function :math:`p^2 \lambda_2`.
    There is some question of whether this with charge screening is
    actually justified or not. For charge screening, make the change
    :math:`p^2 \lambda_2(w) \rightarrow \tilde{p}^2 \lambda(\tilde{w})`.

    Args:
        F (int): The type of Fermi function 0 or 1.
        Zd (int): Charge of the daughter nucleus.
        A (int): Mass of the parent/daughter nucleus.
        w (float): The electron energy.

    Returns:
        float
    """

    # This is not programmed for positron decay
    if Zd < 0: raise ValueError(u"Zd < 0 not allowed in p2lambda2.")
    # Warn if Re(w) < 1
    if np.real(w) < 1.0: print(u"WARNING: Re(w) < 1 in p2lambda2.")

    if sc:
        w = w_screen(Zd, w)
        # Energy conservation
        if np.real(w) < 1.0: return 0.0

    # Special case: when p --> 0, the shape should go to 0, so we can safely return 0
    if abs(w-1.0) < EPSILON: return 0.0

    f0 = Fermi(0, Zd, A, w)
    f1 = Fermi(1, Zd, A, w)
    g1 = gam_ke(1,Zd)
    g2 = gam_ke(2,Zd)
    p  = np.sqrt(w**2 - 1)

    return (p**2.0*((f1/f0))*(2.0+g2)/(2.0+2.0*g1))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import ndimage

#function that creates all output for the thesis
def plots(v, dv_dq, rho0, phi, setup, hbar):
    #function for subplots of two functions at different a
    def row_a(x, q, za, free, a_vals, ylabel, filename, ymin=None, ymax=None):
        fig, axs = plt.subplots(1, len(a_vals), figsize=(15, 5), sharey=True)
        axs[0].set_ylabel(ylabel, fontsize=20)
        for i, a in enumerate(a_vals):
            axs[i].plot(q, free(a), 'r', alpha=0.5, label=fr'free')
            axs[i].plot(q, ndimage.gaussian_filter(free(a), sigma=6), 'r', label=fr'free (smoothed)')
            axs[i].plot(x(a, q), za(a), 'b', label=fr'ZA')
            axs[i].set_xlabel(r'Position ($x$)', fontsize=20)
            axs[i].set_ylim(ymin, ymax)
            axs[i].set_title(f'$a = {a:.2f}$', fontsize=20)
            axs[i].grid()
            axs[i].tick_params(axis='both', which='major', labelsize=15)
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(handles[::-1], labels[::-1], loc='upper left', fontsize=15)
        plt.tight_layout()
        plt.savefig(f'thesis/images/{filename}.pdf')
        plt.show()

    #function that creates animation of two functions over a
    def animate(x, q, za_func, free_func, t_vals, ylabel, filename, ymin=None, ymax=None):
        fig, ax = plt.subplots()
        free_line, = ax.plot([], [], label='free', color='r')
        za_line, = ax.plot([], [], label='ZA', color='b')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left')
        ax.set_xlim(-10, 10)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Position ($x$)')
        ax.set_ylabel(ylabel)
        ax.grid()

        def init():
            free_line.set_data([], [])
            za_line.set_data([], [])
            return free_line, za_line

        def update(frame):
            t = t_vals[frame]
            free_line.set_data(q, free_func(t))
            za_line.set_data(x(t, q), za_func(t))
            ax.set_title(f'$a={t:.2f}$')
            return free_line, za_line

        ani = FuncAnimation(fig, update, frames=len(t_vals), init_func=init, blit=True, interval=20)
        ani.save(f'animations/{filename}.mp4', writer='ffmpeg', fps=25)
        plt.show()

    #function that plots initial conditions
    def initial(q, f, ylabel, filename):
        plt.plot(q, f(q), 'b')
        plt.xlabel(r'Position ($x$)', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid()
        plt.savefig(f'thesis/images/{filename}.pdf')
        plt.show()

    #function that calculates the size of the void
    def size(a, dens, xofq):
        f = dens(a)
        mask_plus = (q > 0)
        mask_minus = (q < 0)
        max_plus = xofq(a, q[mask_plus][np.argmax(f[mask_plus])])
        max_minus = xofq(a, q[mask_minus][np.argmax(f[mask_minus])])
        return max_plus - max_minus
    
    #function that regularizes small denominators
    def reg(x, eps=1e-10):
        return np.sqrt(x**2 + eps**2)

    #set up the box
    res = 4096*4
    Lbox = 20
    dx = Lbox / res
    
    kny = np.pi * res / Lbox
    q = (np.arange(res)/res - 0.5) * Lbox
    k = np.fft.fftfreq(res) * 2 * kny
    k[0] = 0.1

    #calculate time and position of shell crossing
    def a_sc(q):
        #solve inf = densZ(a) => 0 = 1 + a * dv_dq(q)
        return -1 / dv_dq(q)
    mask = (q > 1.2)
    if np.any(mask):
        a_sc_val = np.min(a_sc(q[mask]))
        q_sc_val = q[mask][np.argmin(a_sc(q[mask]))]
        print(f'Shell-crossing time: a_sc = {a_sc_val}')
        print(f'Shell-crossing position: q_sc = {q_sc_val}')

    #zeldovich setup
    def xofqZ(a, q):
        return q + a * v(q)
    
    plt.plot(q, xofqZ(2, q), 'b')
    plt.xlabel(r'Lagrangian Position ($q$)')
    plt.ylabel(r'Eulerian Position ($x$)')
    plt.grid()
    plt.savefig(f'thesis/images/xofq_{setup}.pdf')
    plt.show()


    def interpolation(x, xi, fi):
        idx = np.argsort(xi)
        if len(xi) == 0:
            return np.zeros_like(x)
        return np.interp(x, xi[idx], fi[idx], left=0.0, right=0.0)
    
    def densZ(a):
        return rho0(q) / reg(np.abs(1 + a * dv_dq(q)))
    
    #new method
    def densZ_sum(a):
        x_vals = xofqZ(a, q)
        dx_dq = np.gradient(x_vals, q)
        sign = np.where(dx_dq < 0, True, False)

        if np.all(sign == False):
            stream1 = np.ones_like(q, dtype=bool)
            stream2 = np.zeros_like(q, dtype=bool)
            stream3 = np.zeros_like(q, dtype=bool)
            stream4 = np.zeros_like(q, dtype=bool)
            stream5 = np.zeros_like(q, dtype=bool)
        else:
            ds = np.flatnonzero(sign[1:] != sign[:-1]) + 1
            ds = np.concatenate(([0], ds, [len(sign)]))
            streams = []
            for i in range(len(ds)-1):
                stream = np.zeros_like(dx_dq, dtype=bool)
                stream[ds[i]:ds[i+1]] = True
                streams.append(stream)
            
            stream1, stream2, stream3, stream4, stream5 = streams

        rho_x = rho0(q) / reg(np.abs(1 + a * dv_dq(q)))
        rho_stream1 = rho_x[stream1]
        rho_stream2 = rho_x[stream2]
        rho_stream3 = rho_x[stream3]
        rho_stream4 = rho_x[stream4]
        rho_stream5 = rho_x[stream5]

        rho_1 = interpolation(x_vals, x_vals[stream1], rho_stream1)
        rho_2 = interpolation(x_vals, x_vals[stream2], rho_stream2)
        rho_3 = interpolation(x_vals, x_vals[stream3], rho_stream3)
        rho_4 = interpolation(x_vals, x_vals[stream4], rho_stream4)
        rho_5 = interpolation(x_vals, x_vals[stream5], rho_stream5)

        rho = rho_1 + rho_2 + rho_3 + rho_4 + rho_5
        return rho

    #initialize wavefunction
    aini = 0.01

    def psiZeldo(a, hbar):
        psia = (densZ(a))**(1/2) * np.exp(1j * phi(q) / hbar)
        return psia

    psi_ini = psiZeldo(aini, hbar)

    #time evolution operators
    def drift(psi, a0, a1, hbar):
        da = a1 - a0
        return np.fft.ifft(np.exp(-1j/2.0 * k**2 * hbar * da) * np.fft.fft(psi))
    
    def kick(psi, a0, a1, hbar):
        da = a1 - a0
        alpha = reg(np.real(np.abs(psi)**2))
        return np.exp(-1j/2.0 * np.fft.ifft(-k**2 * np.fft.fft(alpha)) * hbar * da / alpha) * psi
    
    #evolution in the free particle approx
    def evolve_free(psi, a0, a1, hbar):
        return drift(psi, a0, a1, hbar)
    
    def evolve_free_n(psi, a0, a1, hbar, n):
        da = (a1 - a0) / n
        psi_new = psi
        for i in range(n):
            psi_new = drift(psi_new, a0 + i * da, a0 + (i + 1) * da, hbar)
        return psi_new
    
    #evolution with the quantum pressure in schrödi
    def evolve_qp(psi, a0, a1, hbar):
        #kick-drift-kick version
        a = (a0 + a1) / 2
        return kick(drift(kick(psi, a0, a, hbar), a0, a1, hbar), a, a1, hbar)
    
    def evolve_qp_n(psi, a0, a1, hbar, n):
        da = (a1 - a0) / n
        psi_new = psi
        for i in range(n):
            a = a0 + (i + 0.5) * da
            psi_new = kick(drift(kick(psi_new, a0 + i * da, a, hbar), a0 + i * da, a0 + (i + 1) * da, hbar), a, a0 + (i + 1) * da, hbar)
        return psi_new
    
    #calculate velocity of wavefunction
    def calc_vel(psi, hbar):
        denom = reg(psi * np.conj(psi))
        result = hbar/2*np.imag(psi*np.fft.ifft(-1j* k*np.fft.fft(np.conj(psi)))-np.conj(psi)*np.fft.ifft( -1j* k * np.fft.fft(psi)))/denom
        return np.real(result)

    #calculate the quantum pressure potential
    def quantum_pressure(psi, hbar):
        alpha = reg(np.abs(psi)**2)
        result = np.fft.ifft(-k**2 * np.fft.fft(alpha)) * hbar**2 / (2*alpha)
        return np.real(result)
    
    ##start of plots of thesis
    a_vals = [1, a_sc_val, 2]
    a_cont = np.linspace(0, 5, 250)

    #plots of initial conditions
    print('Initial density profile:')
    initial(q, rho0, r'Initial Density ($\rho_0$)', f'InitialDens_{setup}')
    print('Initial velocity potential:')
    initial(q, phi, r'Initial Velocity Potential ($\phi_0$)', f'InitialVelPot_{setup}')
    print('Initial velocity:')
    initial(q, v, r'Initial Velocity ($v_0$)', f'InitialVel_{setup}')

    #density comparison multi streaming
    print('Density comparison multi-streaming:')
    ms = lambda a: densZ(a)
    free = lambda a: np.abs(evolve_free(psi_ini, aini, a, hbar))**2
    row_a(xofqZ, q, ms, free, a_vals, r'Density ($\rho$)', f'MultStream_{setup}', ymin=-1, ymax=15)
    animate(xofqZ, q, ms, free, a_cont, r'Density ($\rho$)', f'MultStream_{setup}', ymin=-1, ymax=15)

    #density comparison correct sum
    print('Density comparison correct sum:')
    za = lambda a: densZ_sum(a)
    row_a(xofqZ, q, za, free, a_vals, r'Density ($\rho$)', f'DensComp_{setup}', ymin=-1, ymax=15)
    animate(xofqZ, q, za, free, a_cont, r'Density ($\rho$)', f'DensComp_{setup}', ymin=-1, ymax=15)

    #denstiy comparison qp
    print('Density comparison quantum pressure:')
    qp = lambda a: np.abs(evolve_qp_n(psi_ini, aini, a, hbar, n=1))**2
    row_a(xofqZ, q, za, qp, a_vals, r'Density ($\rho$)', f'DensCompQP_{setup}', ymin=-1, ymax=15)
    animate(xofqZ, q, za, qp, a_cont, r'Density ($\rho$)', f'DensCompQP_{setup}', ymin=-1, ymax=15)

    #density difference
    print('Density difference:')
    rpd = lambda a: (free(a)-qp(a)) * 2 / (free(a) + qp(a))
    fig, axs = plt.subplots(1, len(a_vals), figsize=(15, 5), sharey=True)
    axs[0].set_ylabel(r'Density Difference ($\frac{\rho_{free}-\rho_{qp}}{\frac{1}{2}(\rho_{free} + \rho_{qp})}$)', fontsize=20)
    for i, a in enumerate(a_vals):
        axs[i].plot(q, rpd(a), 'b')
        axs[i].set_xlabel(r'Position ($x$)', fontsize=20)
        axs[i].set_title(f'a = {a:.2f}', fontsize=20)
        axs[i].grid()
        axs[i].tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(f'thesis/images/DensDiff_{setup}.pdf')
    plt.show()

    #size of the void
    print('Size of the void:')
    a_size = np.linspace(aini, 2, 100)
    q_func = lambda a, q: q
    plt.plot(a_size, np.vectorize(size)(a_size, za, xofqZ), 'b', label='ZA')
    plt.plot(a_size, np.vectorize(size)(a_size, free, q_func), 'r',label='free' )
    plt.axvline(a_sc_val, color='green', linestyle='--', label=fr'Shell-crossing time $a_{{sc}} = {a_sc_val:.2f}$')
    plt.xlabel(r'Scale factor ($a$)', fontsize=14)
    plt.ylabel(r'Size of the void ($L$)', fontsize=14)
    plt.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(f'thesis/images/VoidSize_ZA_{setup}.pdf')
    plt.show()

    #quantum pressure relevance
    print('Quantum pressure relevance:')
    a_qp = np.linspace(a_sc_val-0.5, a_sc_val+0.5, 100)
    diff = []
    pos = []
    for i in a_qp:
        frac = np.abs(rpd(i))
        diff.append(np.max(frac))
        pos.append(xofqZ(i, q)[np.argmax(frac)])
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(a_qp, pos, '.b', label='Position of max. density difference')
    axs[0].axvline(a_sc_val, color='r', linestyle='--', label=fr'Shell-crossing time $a_{{sc}} = {a_sc_val:.2f}$')
    axs[0].set_ylabel(r'Position ($x$)', fontsize=16)
    axs[0].legend(fontsize=12)
    axs[0].tick_params(axis='both', which='major', labelsize=12)
    axs[1].plot(a_qp, diff, 'b', label='Max. density difference')
    axs[1].axvline(a_sc_val, color='r', linestyle='--', label=fr'Shell-crossing time $a_{{sc}} = {a_sc_val:.2f}$')
    axs[1].set_xlabel(r'Scale factor ($a$)', fontsize=16)
    axs[1].set_ylabel(r'Max. density difference ($\max\left[\frac{|\rho_{sc}-\rho_{qp}|}{\frac{1}{2}(\rho_{sc} + \rho_{qp})}\right]$)', fontsize=16)
    axs[1].legend(fontsize=12)
    axs[1].tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(f'thesis/images/QP_relevance_{setup}.pdf')
    plt.show()

    #velocity comparison
    print('Velocity comparison:')
    za = lambda a: v(q)
    free = lambda a: calc_vel(evolve_free(psi_ini, aini, a, hbar), hbar)
    row_a(xofqZ, q, za, free, a_vals, r'Velocity ($v$)', f'VelComp_{setup}', ymin=-1.6, ymax=1.6)
    animate(xofqZ, q, za, free, a_cont, r'Velocity ($v$)', f'VelComp_{setup}', ymin=-1.6, ymax=1.6)

    #Quantum pressure potential
    print('Quantum pressure potential:')
    color = ['g', 'r', 'b']
    for i, a in enumerate(a_vals[::-1]): 
        plt.plot(q, quantum_pressure(evolve_free(psi_ini, aini, a, hbar), hbar), label=f'$a={a:.2f}$', color=color[i])
    plt.xlabel(r'Position ($x$)', fontsize=14)
    plt.ylabel(r'Quantum pressure ($P$)', fontsize=14)
    plt.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=12)
    plt.ylim(-10, 100)
    plt.savefig(f'thesis/images/QP_potential_{setup}.pdf')
    plt.show()

#initial void rho0
def v(q):
    Delta = 2/(np.log(2+np.sqrt(3)))
    sech = lambda x: 1/np.cosh(x)
    A = 1.5*Delta/(2*sech(1/Delta)**2 * np.tanh(1/Delta))
    return 2*A* sech(q/Delta)**2 * np.tanh(q/Delta)/Delta

def dv_dq(q):
    Delta = 2/(np.log(2+np.sqrt(3)))
    sech = lambda q: 1/np.cosh(q)
    A = 1.5*Delta/(2*sech(1/Delta)**2 * np.tanh(1/Delta))
    return 2*A* sech(q/Delta)**2 * (sech(q/Delta)**2 - 2*np.tanh(q/Delta)**2) / Delta**2

def rho0(q):
    A = 0.5
    qc = 1.0
    Delta = 0.1
    dlin = 1 - A * (np.tanh((q + qc)/Delta) - np.tanh((q - qc)/Delta))
    return dlin / np.mean(dlin)

def phi(x):
    Delta = 2/(np.log(2+np.sqrt(3)))
    sech = lambda x: 1/np.cosh(x)
    A = 1.5*Delta/(2*sech(1/Delta)**2 * np.tanh(1/Delta))
    return -A * sech(x/Delta)**2

plots(v, dv_dq, rho0, phi, 'void', hbar=1e-3)

#inital homogenous
def v(q):
    Delta = 2/(np.log(2+np.sqrt(3)))
    sech = lambda x: 1/np.cosh(x)
    A = 1.5*Delta/(2*sech(1/Delta)**2 * np.tanh(1/Delta))
    return 2*A* sech(q/Delta)**2 * np.tanh(q/Delta)/Delta

def dv_dq(q):
    Delta = 2/(np.log(2+np.sqrt(3)))
    sech = lambda q: 1/np.cosh(q)
    A = 1.5*Delta/(2*sech(1/Delta)**2 * np.tanh(1/Delta))
    return 2*A* sech(q/Delta)**2 * (sech(q/Delta)**2 - 2*np.tanh(q/Delta)**2) / Delta**2

def rho0(q):
    return np.ones_like(q)

def phi(x):
    Delta = 2/(np.log(2+np.sqrt(3)))
    sech = lambda x: 1/np.cosh(x)
    A = 1.5*Delta/(2*sech(1/Delta)**2 * np.tanh(1/Delta))
    return -A * sech(x/Delta)**2

plots(v, dv_dq, rho0, phi, 'homogeneous', hbar=1e-3)

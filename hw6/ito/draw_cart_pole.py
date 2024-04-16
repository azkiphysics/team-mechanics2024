import numpy as np

from answer1 import CartPoleEnv


if __name__ == "__main__":
    t_max = 10.0
    dt = 1e-3
    m_cart = 1.0
    m_ball = 1.0
    l_pole = 1.0
    initial_t = 0.0
    initial_x = np.array([0.0, 0.0, 1.0, np.pi / 2 - 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    env = CartPoleEnv(t_max, dt=dt, m_cart=m_cart, m_ball=m_ball, l_pole=l_pole)
    env.reset(initial_t, initial_x)
    env.render()
    env.ax.plot([env.x[0]] * 2, [0, -0.2], linestyle="dashed", color="black")
    env.ax.plot([env.x[1]] * 2, [env.x[2], -0.2], linestyle="dashed", color="black")
    env.ax.plot([env.x[0] + 0.3, -0.6], [0] * 2, linestyle="dashed", color="black")
    env.ax.plot([env.x[1], -0.6], [env.x[2]] * 2, linestyle="dashed", color="black")
    env.ax.plot([-0.6] * 2, [-0.4, env.x[2] + 0.2], color="black")
    env.ax.plot(0.22 * np.cos(np.linspace(0, env.x[3], 10)), 0.22 * np.sin(np.linspace(0, env.x[3], 10)), color="black")
    texts = [
        {"pos": env.x[[1, 2]] / 2.0 + np.array([-0.23, 0.0]), "text": "$l_{\\mathrm{pole}}$"},
        {"pos": env.x[[1, 2]] + np.array([0.12, 0.0]), "text": "$m_{\\mathrm{ball}}$"},
        {"pos": np.array([-0.42, 0.08]), "text": "$m_{\\mathrm{cart}}$"},
        {"pos": np.array([env.x[0] - 0.05, -0.3]), "text": "$x_{\\mathrm{cart}}$"},
        {"pos": np.array([env.x[1] - 0.05, -0.3]), "text": "$x_{\\mathrm{ball}}$"},
        {"pos": np.array([-0.72, -0.05]), "text": "$O$"},
        {"pos": np.array([-0.82, env.x[2] - 0.03]), "text": "$y_{\\mathrm{ball}}$"},
        {
            "pos": 0.22 * np.array([np.cos(env.x[3] / 1.4), np.sin(env.x[3] / 1.4)]),
            "text": "$\\theta_{\\mathrm{pole}}$",
        },
    ]
    for text in texts:
        env.ax.text(*text["pos"], text["text"], color="black", fontsize=25)
    env.ax.set_xlim(-0.8 * env.l_pole, 1.2 * env.l_pole)
    env.ax.set_ylim(-0.35 * env.l_pole, 1.1 * env.l_pole)
    env.fig.suptitle("Fig. 1: Settings of cart pole", fontsize=25, y=0.1)
    env.fig.savefig("cart_pole.png")

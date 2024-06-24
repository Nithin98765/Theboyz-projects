import numpy as np

def initialize_particles(Nt, Nv):
    """
    Initializes particles for MCT-PSO.

    Args:
        Nt (int): Number of tasks.
        Nv (int): Number of VMs.

    Returns:
        list: List of particles, each represented as a binary vector.
    """
    num_particles = 10  # You can adjust this as needed
    particles = []
    for _ in range(num_particles):
        particle = np.random.randint(2, size=(Nt, Nv))
        particles.append(particle)
    return particles

def calculate_completion_time(particle, T, V):
    """
    Calculates the total completion time for a particle.

    Args:
        particle (np.ndarray): Binary vector representing task allocation.
        T (list): Task lengths (execution times) for each task (length Nt).
        V (list): VMs' execution rates (length Nv).

    Returns:
        float: Total completion time.
    """
    execution_times = np.dot(T, particle) / V
    return np.max(execution_times)  # Assuming maximization problem

def update_velocity(velocity, particle, global_best_particle, w=0.7, c1=1.5, c2=1.5):
    """
    Updates the velocity of a particle.

    Args:
        velocity (np.ndarray): Current velocity.
        particle (np.ndarray): Current particle.
        global_best_particle (np.ndarray): Global best particle.
        w (float): Inertia weight.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.

    Returns:
        np.ndarray: Updated velocity.
    """
    r1, r2 = np.random.rand(), np.random.rand()
    cognitive_component = c1 * r1 * (particle - particle)
    social_component = c2 * r2 * (global_best_particle - particle)
    new_velocity = w * velocity + cognitive_component + social_component
    return new_velocity

def mct_pso(Nt, T, Nv, V, num_iterations=100):
    particles = initialize_particles(Nt, Nv)
    global_best_particle = None
    global_best_completion_time = float('inf')

    for it in range(num_iterations):
        for i, particle in enumerate(particles):
            completion_time = calculate_completion_time(particle, T, V)
            if completion_time < global_best_completion_time:
                global_best_completion_time = completion_time
                global_best_particle = particle

            # Update velocity and position
            velocity = np.zeros_like(particle)
            velocity = update_velocity(velocity, particle, global_best_particle)
            particle = np.where(velocity > 0, 1, 0)

    return global_best_particle

# Example usage
Nt = 15
T = [1.27,2.06,0.16,2.29,4.23,0.79,0.72,0.99,7.33,3.08,2.12,10.14,0.14,3.79,5.79]  # Task lengths (execution times)
Nv = 5
V = [5,5,5,5,5]  # VMs' execution rates

allocation_matrix = mct_pso(Nt, T, Nv, V)
for row in allocation_matrix:
    print(row)

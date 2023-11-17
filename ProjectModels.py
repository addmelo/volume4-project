# Volume-4-Project-Models

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

class Solow_Model_Parameters():
    """This class holds the parameters for the Solow Growth Model."""
    
    def __init__(self, A=1, alpha=0.5, delta=0.08, s=0.3, weights=lambda x: (x >= 18) * (x <= 65)):
        """
        Initializes the parameters for the Solow Growth Model.
        
        Parameters:
        - A: Total factor productivity (Number or function with respect to time)
        - alpha: Output elasticity of capital
        - delta: Depreciation rate
        - s: Savings rate  (Number or function with respect to time)
        - weights: Function defining the age distribution weights
        """
        self.A = A
        self.alpha = alpha
        self.delta = delta
        self.s = s
        self.weights = weights
    
    def y(self, k, t, n=None):
        """
        Computes output (Y) given the capital (K), time (t), and labor (N).
        
        Parameters:
        - k: Capital
        - t: Time
        - n: Labor (optional)
        
        Returns:
        - Output (Y)
        """
        if callable(self.A):
            A = self.A(t)
        else:
            A = self.A
        if n is None:
            return A * k ** self.alpha
        else:
            return A * k ** self.alpha * n
    
    def kprime(self, k, dpop, pop, start_working, retire, t):
        """
        Computes the change in capital (K') given the capital (K), population change (dpop),
        total population (pop), start_working age, retire age, and time (t).
        
        Parameters:
        - k: Capital
        - dpop: Change in population
        - pop: Total population
        - start_working: Starting age of the workforce
        - retire: Retirement age
        - t: Time
        
        Returns:
        - Change in capital (K')
        """
        if callable(self.s):
            s = self.s(t)
        else:
            s = self.s
        n = (dpop[start_working] - dpop[retire + 1]) / np.sum(pop[start_working:retire + 1])
        return s * self.y(k, t,np.sum(pop[start_working:retire+1],axis =0)/np.sum(pop,axis = 0)) - (self.delta + n) * k

class Solution():
    """This class holds results from the Ordinary Differential Equation (ODE) solution."""
    
    def __init__(self, t, population, capital=None, y=None):
        """
        Initializes the solution results for the ODE.
        
        Parameters:
        - t: Time
        - population: Total population
        - capital: Capital (optional)
        - y: Output (Y) (optional)
        """
        self.t = t
        self.population = population
        self.capital = capital
        self.y = y




import numpy as np
from scipy.integrate import solve_ivp

class Population_Solow_Model():
    def __init__(self, life_expectancy=85, fertility_rate=2.0, fertility_starts=18, fertility_ends=40, solow_growth_parameters=None):
        """
        Initializes the Population Solow Model with parameters.
        
        Parameters:
        - life_expectancy: The average life expectancy in the model. Must be an integer and at least 40. Default is 85.
        - fertility_rate: A constant fertility rate or a function of time. Default is 2.0.
        - fertility_starts: The age at which fertility begins. Default is 18.
        - fertility_ends: The age at which fertility ends. Default is 40.
        - solow_growth_parameters: Parameters for the Solow Growth Model. Default is None.
        """
        self.fertility_rate = fertility_rate
        
        # Validate input types
        if not isinstance(life_expectancy, int):
            raise TypeError("'life_expectancy' must be an integer in this model.")
        self.life_expectancy = life_expectancy

        if not isinstance(fertility_starts, int):
            raise TypeError("'fertility_starts' must be an integer in this model.")
        self.fertility_starts = fertility_starts

        if not isinstance(fertility_ends, int):
            raise TypeError("'fertility_ends' must be an integer in this model.")
        self.fertility_ends = fertility_ends

        self.ode = None
        self.labels = None
        self.start_working = None
        self.retire = None
        self.SGP = solow_growth_parameters
        self.include_SGP = False
        return
    
    def prep_model(self, start_working=18, retire=65):
        """
        Prepares the population model based on the SIR framework with granular fertility.

        Parameters:
        - start_working: The age at which individuals start working. Default is 18.
        - retire: The retirement age. Default is 65.

        Returns:
        - self: The Population Solow Model instance with prepared settings.
        
        Raises:
        - TypeError: If life_expectancy is not an integer.
        - ValueError: If life_expectancy is less than 40.
        """
        # Validate life expectancy
        if not isinstance(self.life_expectancy, int):
            raise TypeError("Life expectancy must be an integer in this model.")
        elif self.life_expectancy < 40:
            raise ValueError("This model requires life_expectancy to be at least 40.")

        self.include_SGP = self.SGP is not None
        
        # Define the ordinary differential equation (ODE)
        def ode(t, x):
            """
            Ordinary differential equation representing the SIR population dynamics with granular fertility.

            Parameters:
            - t: Time variable.
            - x: Array representing the state variables, where x[i] represents the number of people at age i.

            Returns:
            - dx: Array representing the rates of change of the state variables.
            """
            dx = np.zeros_like(x)

            # Calculate the births based on the fertility rate
            if callable(self.fertility_rate):
                dx[0] = 0.5 * self.fertility_rate(t) * np.mean(x[self.fertility_starts:self.fertility_ends + 1]) - x[0]
            else:
                dx[0] = 0.5 * self.fertility_rate * np.mean(x[self.fertility_starts:self.fertility_ends + 1]) - x[0]

            # Calculate the change in other age categories
            dx[1:] = x[:-1] - x[1:]

            # Add the capital allocation variables if necessary
            if self.include_SGP:
                dx[-1] = self.SGP.kprime(x[-1], dx[:-1], x[:-1], start_working, retire, t)

            return dx

        # Set attributes for the model
        self.ode = ode
        self.labels = [f"{i}-{i+1}" for i in range(self.life_expectancy)]
        self.start_working = start_working
        self.retire = retire
        return self
    
    def solve(self, t_points, starting_population, starting_capital=None):
        """
        Solves the ODE for population dynamics given initial conditions.

        Parameters:
        - t_points: Time points to solve the ODE.
        - starting_population: Initial population distribution across age categories.
        - starting_capital: Initial capital (if Solow Growth Parameters are included).

        Returns:
        - Solution: Instance of the Solution class containing the ODE results.
        
        Raises:
        - AssertionError: If Population_Solow_Model().ode is not defined.
        """
        if self.ode is None:
            raise AssertionError("'Population_Solow_Model().ode' has not been defined. Run 'Population_Solow_Model().create_model' first.")

        # Prepare initial state
        if not self.include_SGP:
            x = starting_population
        else:
            x = np.concatenate([starting_population, np.array([starting_capital])])

        t_span = (np.min(t_points), np.max(t_points))
        
        # Solve the ODE
        sol = solve_ivp(self.ode, t_span, x, t_eval=t_points)

        if self.SGP is None:
            return Solution(sol.t, sol.y)
        else:
            # Calculate labor and return Solution instance
            n = np.sum(sol.y[:-1] * np.reshape(self.SGP.weights(np.arange(self.life_expectancy)), (-1, 1)), axis=0) / np.sum(sol.y[:-1], axis=0)
            return Solution(sol.t, sol.y[:-1], sol.y[-1], self.SGP.y(sol.y[-1], ts, n))

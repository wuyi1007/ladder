"""
base_questions.py

This file contains a list of base integral problems that are considered challenging.
Each problem is represented as a string which we expect to follow the pattern:
    integrate(<integrand>, x)
For example: "integrate(1/(x**2 - x + 1), x)"
"""

#2020 - 2024 MIT intergration bee questions
#https://math.mit.edu/~yyao1/integrationbee.html
#just the qualifier tests
BASE_QUESTIONS = [
    "integrate(exp(-x**2)*sin(x), (x, -oo, oo))",
    "integrate((x**2 + 1)*log(1 + x**2), x)",
    "integrate((sin(3*x) - 3*sin(x))/cos(x)**3, x)",
    "integrate(x**3*sqrt(1 + x**2), x)",
    "integrate(log(x)**2/x, (x, 1, E))",
    "integrate((x**4 - 1)/(x**2 + 1), x)",
    "integrate(sin(x)**3*cos(x)**5, x)",
    "integrate(x*exp(x)*cos(x), x)",
    "integrate((1 + tan(x))**2, x)",
    "integrate(1/(x**2*sqrt(1 - 1/x**2)), (x, 2, oo))",
    "integrate((x**2 + 4*x + 5)/((x+1)**2 + 4), x)",
    "integrate(x**2*exp(-x), (x, 0, oo))",
    "integrate((sin(x) + x*cos(x))/x**2, x)",
    "integrate(1/(x**2 - 4*x + 5), x)",
    "integrate((3*x**2 - 4*x + 1)*sqrt(x), x)",
    "integrate((x*sin(x) + cos(x))/x, x)",
    "integrate((1 - cos(2*x))/x, (x, 0, pi))",
    "integrate((log(x))**3, x)",
    "integrate((x**2 + 1)/(x**4 + 1), x)",
    "integrate(sin(x)/(1 + cos(x)), x)",
    "integrate(Abs(x - 1)*(x + 2), (x, -2, 3))",
    "integrate((x - floor(x))**2, (x, 0, 10))",
    "integrate(max(sin(x), 0), (x, 0, 2*pi))",
    "integrate((x - floor(x))*(1 + floor(x)), (x, 0, 5))",
    "integrate((sin(5*x)/sin(x)), (x, 0, pi/2))",
    "integrate((x**3 + 1)/(x**2 + 1), x)",
    "integrate((x**2 - 1)/(x**4 + 1), (x, 0, oo))",
    "integrate(exp(-x)*sin(x), (x, 0, oo))",
    "integrate(cos(x)**2/log(x), (x, 2, E**2))",
    "integrate(x*sqrt(4 - x**2), (x, 0, 2))",
    "integrate((3 + 2*cos(x))/ (5 + 4*sin(x)), x)",
    "integrate((x**2+9)/(x**4+6*x**2+25), x)",
    "integrate(tan(x)**3, x)",
    "integrate((1 + x**4)*exp(-x**2), (x, 0, oo))",
    "integrate((2*x)/(1 + x**4), x)",
    "integrate((x**2 - 4)*exp(x), (x, 0, 2))",
    "integrate(sinh(x)/(1 + cosh(x)), x)",
    "integrate(x**2*log(x)/(1 + x**3), (x, 0, 1))",
    "integrate((sin(x) - x*cos(x))/x**2, x)",
    "integrate(1/sqrt(1 + x**4), (x, 0, 1))"
  ]

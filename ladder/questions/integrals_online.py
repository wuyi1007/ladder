"""
base_questions.py

This file contains a list of base integral problems that are considered challenging.
Each problem is represented as a string which we expect to follow the pattern:
    integrate(<integrand>, x)
For example: "integrate(1/(x**2 - x + 1), x)"
"""

#GPT 4o-mini 51% accuracy


BASE_QUESTIONS = [
    "integrate(1/(x + 1 + 2*sqrt(x))**2, (x, 0, oo))",
    "integrate(x**2*cos(arccsc(x)), x)",
    "integrate(sum((2025*x)**k*exp(-2025*x)/factorial(k), (k, 1, oo)), (x, 0, 1/2025))",
    "integrate(1/(1 - x**4), x)",
    "integrate(cos(20*x)*cos(25*x), (x, -pi/2, pi/2))",
    "integrate(max(x, x**2, x**3), (x, -2, 2))",
    "integrate(sqrt(x), (x, 0, 2025))",
    "integrate((cos(x)**4 - 1)/sin(x)**8, x)",
    "integrate(sin(x)*sin(sin(x))*sin(cos(x)) + cos(x)*cos(sin(x))*cos(cos(x)), x)",
    "integrate(exp(2*x)*sin(sinh(x)), (x, -1, 1))",
    "integrate((x + sqrt(x))/(1 + sqrt(x)), x)",
    "integrate(exp(x+1)/(exp(x) + 1), x)",
    "integrate(3*sin(x) - sin(3*x), x)",
    "integrate(cos(20*x)*sin(25*x), (x, -pi/2, pi/2))",
    "integrate(sin(x)*cos(x)*tan(x)*cot(x)*sec(x)*csc(x), (x, 0, 2*pi))",
    "integrate(x**2024*(1 - x**2025)**2025, (x, 0, 1))",
    "integrate(cos(x) - 3/x**5, x)",
    "integrate(cos(x) - 3/x**5, (x, -3, 4))",
    "integrate(12*x**3 - 9*x**2 + 2, (x, 1, 6))",
    "integrate(8/sqrt(t) - 12*sqrt(t**3), (t, 1, 4))",
    "integrate(1/(7*z) + cbrt(z**2)/4 - 1/(2*z**3), (z, 1, 2))",
    "integrate(x**6 - x**4 + 1/x**2, (x, -2, 4))",
    "integrate(7*sin(t) - 2*cos(t), (t, 0, pi/2))",
    "integrate(sec(z)*tan(z) - 1, (z, 0, pi))",
    "integrate(2*sec(w)**2 - 8*csc(w)*cot(w), (w, pi/6, pi/3))",
    "integrate(exp(x) + 1/(x**2 + 1), (x, 0, 2))"
  ] 

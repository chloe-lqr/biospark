import ast
import functools
import operator as op

__all__ = ['eval_expr']

# supported operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}

# def power(a, b):
#     # limit input arguments for a**b
#     if any(abs(n) > 100 for n in [a, b]):
#         raise ValueError((a,b))
#     return op.pow(a, b)
# operators[ast.Pow] = power

def limit(max_=None):
    # Return decorator that limits allowed returned values
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            try:
                mag = abs(ret)
            except TypeError:
                pass # not applicable
            else:
                if mag > max_:
                    raise ValueError(ret)
            return ret
        return wrapper
    return decorator

def eval_(node):
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)

@limit(max_=10**100)
def eval_expr(expr):
    # for parsing simple expressions, such as exponents
    # lifted from http://stackoverflow.com/a/9558001/425458
    return eval_(ast.parse(expr, mode='eval').body)
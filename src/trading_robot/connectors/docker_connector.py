import mt5linux  as MetaTrader5

mt5 = MetaTrader5.MetaTrader5(host='localhost', port=8001)

# def connecting_docker_mt5(host='localhost', port=18812):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             mt5 = MetaTrader5.MetaTrader5(host='localhost', port=8001)
            
#             if not mt5.initialize():
#                 print(f"Failed to connect to MetaTrader 5 at {host}:{port}")
#                 return None
#             try:
#                 result = func(*args, **kwargs)
#             finally:
#                 mt5.shutdown()
#             return result
#         return wrapper
#     return decorator
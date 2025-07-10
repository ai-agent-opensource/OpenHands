 # 예시: 실제 호출 체인 추적
def log_call_chain(func_name, *args, **kwargs):
        print(f"CALL: {func_name}")
        print("--- args ---")
        for arg in args:
            print(arg)
        print("--- kwargs ---")
        for k, v in kwargs.items():
            print(f"{k}: {v}")
        return func_name, args, kwargs

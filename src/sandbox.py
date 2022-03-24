class HelloWorld:
    def __init__(self):
        pass

    @staticmethod
    def f1(a = None):
        print(a)
    
    def f2(self):
        self.f1("Hello ss")
        print("world")
    
if __name__ == '__main__':
    hello = HelloWorld()
    hello.f2()
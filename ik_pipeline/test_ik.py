from ik_pipeline.ik_controller import IKController

controller = IKController()

while True:
    x = float(input("Target X: "))
    y = float(input("Target Y: "))

    result = controller.compute((0, 0), (x, y))
    print("Result:", result)

 if action.max().item() > 2.0: 
                print(action)
                print("Found something greater than 1.5")
            if action.max().item() > 3.0:
                print("oh no found something too big")
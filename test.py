answer=0
def test():
    global answer
    answer+=1
    return 0
test()
test()
print(answer)
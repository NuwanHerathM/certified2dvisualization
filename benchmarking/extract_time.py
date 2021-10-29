def string2seconds(str):
    min, tail = str.split("\t")[1].split("m")
    sec, millisec = tail.split("s")[0].split(",")
    return int(min) * 60 + int (sec) + int(millisec) * .001

with open("tortue") as file:
    file.readline()
    file.readline()
    user = string2seconds(file.readline())
    sys = string2seconds(file.readline())
    print(f"{user + sys:.3f}")


#########
# Class #
#########
class ImgAveraging:
    def __init__(self, nb_max):
        self.__NB_MAX_REF = nb_max
        self.__NB_MAX = nb_max
        self.__nb_stock = 0
        self.__first = 0
        self.__last = 0
        self.__container = [0 for _ in range(nb_max)]
        self.__img_sum = 0

    def average(self):
        av = np.array([[0]])

        if self.__nb_stock > 0:
            av = self.__img_sum / self.__nb_stock

        return av

    def new_capacity(self, nb):
        # redefined object
        if self.__NB_MAX_REF >= nb > 0:
            self.__NB_MAX = nb
            self.__nb_stock = 0
            self.__first = 0
            self.__last = 0
            self.__img_sum = 0

    def new_img(self, img):
        if self.__nb_stock == self.__NB_MAX:
            self.__img_sum -= self.__peek()

        self.__img_sum += img
        self.__push(img)

    def __peek(self):
        item = None

        if self.__nb_stock > 0:
            item = self.__container[self.__first]

        return item

    def __pop(self):
        item = None

        if self.__nb_stock > 0:
            item = self.__container[self.__first]
            self.__first += 1
            self.__first %= self.__NB_MAX
            self.__nb_stock -= 1

        return item

    def __push(self, img):
        # erase first item if the container is full
        if self.__nb_stock == self.__NB_MAX:
            self.__pop()

        self.__container[self.__last] = img
        self.__last += 1
        self.__last %= self.__NB_MAX
        self.__nb_stock += 1

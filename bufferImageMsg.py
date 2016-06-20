import rospy
from sensor_msgs.msg import Image


class BufferImageMsg:
    def __init__(self, nb_max):
        if nb_max < 1:
            print("\nError: not enough capacity!\n")
            return

        self.__NB_MAX_REF = nb_max
        self.__NB_MAX = 1
        self.__nb_stock = 0
        self.__first = 0
        self.__last = 0
        self.__container = [0 for _ in range(nb_max)]

        self.__last_msg = None

    def set_subscriber(self, channel):
        rospy.Subscriber(channel, Image, self.__push)

    def run(self, capacity=1):
        if capacity < 1:
            print("\nError: not enough capacity!\n")
            return

        self.__NB_MAX = min(capacity, self.__NB_MAX_REF)
        self.__nb_stock = 0
        self.__first = self.__last

    def get_img_msg(self):
        return self.__pop()

    def get_last_img_msg(self):
        msg = self.__last_msg
        self.__last_msg = None
        return msg

    def get_nb(self):
        return self.__nb_stock

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

    def __push(self, msg):
        # erase first item if the container is full
        if self.__nb_stock == self.__NB_MAX:
            self.__pop()

        self.__container[self.__last] = msg
        self.__last_msg = msg
        self.__last += 1
        self.__last %= self.__NB_MAX
        self.__nb_stock += 1

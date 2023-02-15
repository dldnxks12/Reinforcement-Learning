"""

Python에 내장된 property() 함수와 @property 데코레이터에 대해 알아보자.

*reference : https://www.daleseo.com/python-property/

"""

class Person1:
    def __init__(self, first_name, last_name, age):

        # 3개의 field
        self.first_name = first_name
        self.last_name  = last_name
        self.age        = age


# Person class의 instance를 만들고, 위 3개의 field를 읽거나 써보자.


person_instance = Person1("lee", "jongsoo", 29)

print(person_instance.first_name)
print(person_instance.last_name)
print(person_instance.age)

person_instance.age = person_instance.age + 1 # -> 이렇게 class 내부 field에 접근해서 데이터를 바꿀 수 있다.
print(person_instance.age)

"""

위에서 처럼 class 내부 field를 바꾸는 건 개발자 입장에서 생각보다 바람직하지 않다. 
편하긴 편할테지만 다른 사용자가 막 바꾸도록 놔두는 건 사실 말이 안된다. 

"""

# Getter / Setter -> Getter : 데이터를 읽어주는 메소드, Setter : 데이터를 바꿔주는 메소드
# Person2 class 내부에 get_age , set_age 함수를 만들자 -> getter setter 정의
class Person2:
    def __init__(self, first_name, last_name, age):

        # 3개의 field
        self.first_name = first_name
        self.last_name  = last_name
        self.set_age(age)  # class instance 생성할 때 setter 함수를 호출하면서 age 값 설정

    def get_age(self):
        return self._age

    def set_age(self, age):
        if age < 0 :
            raise ValueError("Invalid age")
        self._age = age # 외부에서 직접 접근하지 못하도록 age 대신 _age라는 이름으로 선언 (외부에서 접근하지 않는 필드에 대한 관행)

person2 = Person2('j','s', 29)

print(person2.get_age())  # 29

person2.set_age(person2.get_age()+1) # _age 변수 조작

print(person2.get_age())  # 30

"""

위처럼 내부적으로 직접 건드리지 못하는 변수(_age)를 만들 수 있고, 함수 getter, setter를 통해 이 변수를 조작한다. 

즉, 내부 _age 데이터에 접근하려면 반드시 getter , setter 메서드를 통해야 한다. 

이렇게 하면 class 내부의 데이터에 대한 접근을 어느정도 통제할 수 있게 됬지만 코드가 좀 지저분하다.

파이썬 내장 함수인 property() 를 사용하면 필드명을 사용하는 것처럼 깔끔하게 getter/setter 메서드가 호출되게 할 수 있다.

다음을 이해해보자.

"""

class Person3:
    def __init__(self, first_name, last_name, age):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age

    def get_age(self):
        return self._age # self.age != self._age

    def set_age(self, age):
        if age < 0 :
            raise ValueError("Error")
        self._age = age # 직접 접근하지 못하는 내부 변수 _age

    age = property(get_age, set_age)  # property(getter, setter) -> age라는 필드명을 이용해서 다시 나이 데이터에 접근할 수 있다.

person3 = Person3('j', 's', 29)

person3.age = 1
print(person3.age)
person3.age = -1
print(person3.age) # Error !
print(person3.age + 1)

"""

그냥 보면 getter / setter 메서드를 통하지 않고 그냥 바로 field에 접근하는 것 같지만 
내부적을 getter / setter 메소드를 통한다. -> 따라서 -1을 할당 했을 때 ValueError가 일어난다.  


@property 데코레이터를 이용하면 위랑 똑같이 작동하는 코드를 조~금 더 간결하게 그리고 읽기 편하게 작성할 수 있다.

"""


class Person4:
    def __init__(self, first_name, last_name, age):
        self.first_name = first_name
        self.last_name = last_name
        self.age = age

    # 기존의 getter method
    @property
    def age(self):
        return self._age

    # 기존의 setter method <field 이름>.setter
    @age.setter
    def age(self, age):
        if age < 0:
            raise ValueError("Error")
        self._age = age


person4 = Person4('j', 's', 29)
print(person4.age)

person4.age = 1
print(person4.age)
person4.age = -1
print(person4.age)






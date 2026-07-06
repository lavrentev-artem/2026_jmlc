from dataclasses import dataclass
from abc import ABC
from uuid import UUID
from datetime import datetime



@dataclass
class Person(ABC):
    """
    Basic class representing a person.

    Attributes:
        id(int): person id
        created_at(datetime): person instance creation date and time
        status(str): person status
    """
    id: UUID
    created_at: datetime
    status: str
    group: str


@dataclass
class User(Person):
    """
    Commercial user.
    
    Attributes:
        first_name(str): user's first name
        last_name(str): user's last name
        gender(str): user's gender
        birthday(str): user's birthday
        country(str): user's country
        group(str): user's group
    """
    first_name: str
    last_name: str
    gender: str
    birthday: str
    country: str
    group: str = 'user'


@dataclass
class Admin(Person):
    """
    Administrative user.

    Attributes:
        first_name(str): admin's first name
        last_name(str): admin's last name
        group(str): admin's group
    """
    first_name: str
    last_name: str
    group: str = 'admin'


@dataclass
class OperationInput:
    """
    Input of an operation.

    Attributes:
        operation_input: str
    """
    operation_input: str


@dataclass
class OperationOutput:
    """
    Output of an operation.

    Attributes:
        operation_output: str
    """
    operation_output: str


@dataclass
class OperationCost:
    """
    Cost for an operation.

    Attributes:
        operation_cost: float
    """
    operation_cost: float


@dataclass
class Operation:
    """
    User's operation.

    Attributes:
         id(int): operation id
         operation_datetime(datetime): operation datetime
         input(OperationInput): user's request
         output(OperationOutput): system's response
         cost(OperationCost): cost of operation
    """
    id: int
    operation_datetime: datetime
    input: OperationInput | None
    output: OperationOutput | None
    cost: OperationCost


@dataclass
class Balance:
    """
    User's balance in credits.

    Attributes:
        amount(float): balance amount
    """
    amount: float


@dataclass
class Charge:
    """
    Charge for an operation.

    Attributes:
        id(UUID): charge id
        amount(float): charge amount
        charge_datetime(datetime): charge creation date and time
    """
    id: UUID
    amount: float
    charge_datetime: datetime


@dataclass
class Top_Up:
    """
    Top up operation.

    Attributes:
        id(UUID): top up id
        amount(float): top up amount
        top_up_datetime(datetime): top up creation date and time
    """
    id: UUID
    amount: float
    top_up_datetime: datetime
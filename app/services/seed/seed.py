from sqlmodel import Session
from models.user import UserCreate
from services.crud.crud_user import create_user, create_admin
from services.crud.crud_balance import create_balance_transaction
from services.crud.crud_operation import create_operation

from models.balance import BalanceTransactionCreate, BalanceTransactionType
from models.operation import OperationCreate, OperationType, OperationStatus


def seed_data(session: Session):
    """
    Seeds the database with test data
    Args:
        session: DB session
    Returns: None
    """
    with session.no_autoflush:
        #-- User 1

        user1_in = UserCreate(
            full_name = "Richard Feynman",
            email = "richard.feynman@test.data",
            password = "richard.feynman.password"
        )
        user1_db = create_user(user1_in, session)

        user1_topup1_in = BalanceTransactionCreate(
            user_id = user1_db.id,
            transaction_amount = 100.00,
            transaction_type = BalanceTransactionType.TOPUP,
            external_id = "elvb-3598-sdfj-2330",
            description = "Test topup"
        )
        create_balance_transaction(user1_topup1_in, session)

        user1_operation1_in = OperationCreate(
            user_id = user1_db.id,
            type = OperationType.PROMPT,
            status = OperationStatus.COMPLETE,
            cost = "1.00",
            operation_input = "What is the code of the Oppenheimer's safe?",
            operation_output = "0000"
        )
        create_operation(user1_operation1_in, session)

        user1_charge1_in = BalanceTransactionCreate(
            user_id = user1_db.id,
            transaction_amount = -1.00,
            transaction_type = BalanceTransactionType.CHARGE,
            description = "Charge for test prompt"
        )
        create_balance_transaction(user1_charge1_in, session)


        #-- User 2

        user2_in = UserCreate(
            full_name="Niels Bor",
            email="niels.bor@test.data",
            password="niels.bor.password"
        )
        user2_db = create_user(user2_in, session)

        user2_topup1_in = BalanceTransactionCreate(
            user_id = user2_db.id,
            transaction_amount = 30.00,
            transaction_type = BalanceTransactionType.TOPUP,
            external_id = "yght-7618-pmrz-3677",
            description = "Test topup"
        )
        create_balance_transaction(user2_topup1_in, session)

        user2_operation1_in = OperationCreate(
            user_id = user2_db.id,
            type = OperationType.PROMPT,
            status = OperationStatus.COMPLETE,
            cost = "1.00",
            operation_input = "How should I depict an atom's model?",
            operation_output = "Draw it like a ball"
        )
        create_operation(user2_operation1_in, session)

        user2_charge1_in = BalanceTransactionCreate(
            user_id = user2_db.id,
            transaction_amount = -1.00,
            transaction_type = BalanceTransactionType.CHARGE,
            description = "Charge for test prompt"
        )
        create_balance_transaction(user2_charge1_in, session)

        user2_operation2_in = OperationCreate(
            user_id = user2_db.id,
            type = OperationType.PROMPT,
            status = OperationStatus.COMPLETE,
            cost="1.00",
            operation_input = "No, it will not work. Maybe as a planetary system?",
            operation_output = "As you wish. They will not be able to check it anyway."
        )
        create_operation(user2_operation2_in, session)

        user2_charge2_in = BalanceTransactionCreate(
            user_id=user2_db.id,
            transaction_amount=-1.00,
            transaction_type=BalanceTransactionType.CHARGE,
            description="Charge for test prompt"
        )
        create_balance_transaction(user2_charge2_in, session)


        #-- User 3

        user3_in = UserCreate(
            full_name="Enrico Fermi",
            email="enrico.fermi@test.data",
            password="enrico.fermi.password"
        )
        user3_db = create_user(user3_in, session)

        user3_topup1_in = BalanceTransactionCreate(
            user_id = user3_db.id,
            transaction_amount = 1.00,
            transaction_type = BalanceTransactionType.TOPUP,
            external_id = "ucmr-3895-pdfk-9983",
            description = "Test topup"
        )
        create_balance_transaction(user3_topup1_in, session)

        user3_operation1_in = OperationCreate(
            user_id = user3_db.id,
            type = OperationType.PROMPT,
            status = OperationStatus.COMPLETE,
            cost = "1.00",
            operation_input = "Where are all extraterrestrials?",
            operation_output = "42"
        )
        create_operation(user3_operation1_in, session)

        user3_charge1_in = BalanceTransactionCreate(
            user_id=user3_db.id,
            transaction_amount=-1.00,
            transaction_type=BalanceTransactionType.CHARGE,
            description="Charge for test prompt"
        )
        create_balance_transaction(user3_charge1_in, session)


        #-- User 4

        user4_in = UserCreate(
            full_name="1",
            email="1",
            password="1"
        )
        user4_db = create_user(user4_in, session)

        user4_topup1_in = BalanceTransactionCreate(
            user_id = user4_db.id,
            transaction_amount = 10.00,
            transaction_type = BalanceTransactionType.TOPUP,
            external_id = "acrt-3947-enpa-6284",
            description = "Test topup"
        )
        create_balance_transaction(user4_topup1_in, session)

        user4_operation1_in = OperationCreate(
            user_id = user4_db.id,
            type = OperationType.PROMPT,
            status = OperationStatus.COMPLETE,
            cost = "1.00",
            operation_input = "Test prompt",
            operation_output = "Test response"
        )
        create_operation(user4_operation1_in, session)

        user4_charge1_in = BalanceTransactionCreate(
            user_id=user4_db.id,
            transaction_amount=-1.00,
            transaction_type=BalanceTransactionType.CHARGE,
            description="Charge for test prompt"
        )
        create_balance_transaction(user4_charge1_in, session)


        # Admins
        admin1_in = UserCreate(
            full_name="Robert Oppenheimer",
            email="robert.oppenheimer@test.data",
            password="robert.oppenheimer.password"
        )
        admin1_db = create_admin(admin1_in, session)

        admin1_topup1_in = BalanceTransactionCreate(
            user_id = admin1_db.id,
            transaction_amount = 1.00,
            transaction_type = "topup",
            external_id = "urnx-2263-apvn-7143",
            description = "Test topup"
        )
        create_balance_transaction(admin1_topup1_in, session)

        admin1_operation1_in = OperationCreate(
            user_id = admin1_db.id,
            type = OperationType.PROMPT,
            status = OperationStatus.COMPLETE,
            cost = "1.00",
            operation_input = "What did guys do when I was absent?",
            operation_output = "Feynman stole 10$ from your safe"
        )
        create_operation(admin1_operation1_in, session)

        admin1_charge1_in = BalanceTransactionCreate(
            user_id=admin1_db.id,
            transaction_amount=-1.00,
            transaction_type=BalanceTransactionType.CHARGE,
            description="Charge for test prompt"
        )
        create_balance_transaction(admin1_charge1_in, session)


        try:
            session.commit()
        except Exception as e:
            session.rollback()
            raise

        return None
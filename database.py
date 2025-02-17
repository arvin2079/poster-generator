import sqlite3
import numpy as np

conn = sqlite3.connect("roi.db")

cursor = conn.cursor()

# Enable foreign key support (required in SQLite)
cursor.execute("PRAGMA foreign_keys = ON")


def initiate_relations():

    # create posters relation
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS posters(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tree_type TEXT NOT NULL, 
        image_name TEXT NOT NULL,
        CONSTRAINT unique_image_name_tree_type UNIQUE (tree_type, image_name)
    )
    """
    )

    # create gift taker relation
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS gift_takers(
        poster_id INTEGER PRIMARY KEY,
        tlx INTEGER UNSIGNED NOT NULL,
        tly INTEGER UNSIGNED NOT NULL,
        trx INTEGER UNSIGNED NOT NULL,
        try INTEGER UNSIGNED NOT NULL,
        brx INTEGER UNSIGNED NOT NULL,
        bry INTEGER UNSIGNED NOT NULL,
        blx INTEGER UNSIGNED NOT NULL,
        bly INTEGER UNSIGNED NOT NULL,
        FOREIGN KEY (poster_id) REFERENCES posters(id) ON DELETE CASCADE
    )
    """
    )

    # create information box relation
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS information_boxes(
        poster_id INTEGER PRIMARY KEY,
        tlx INTEGER UNSIGNED NOT NULL,
        tly INTEGER UNSIGNED NOT NULL,
        trx INTEGER UNSIGNED NOT NULL,
        try INTEGER UNSIGNED NOT NULL,
        brx INTEGER UNSIGNED NOT NULL,
        bry INTEGER UNSIGNED NOT NULL,
        blx INTEGER UNSIGNED NOT NULL,
        bly INTEGER UNSIGNED NOT NULL,
        FOREIGN KEY (poster_id) REFERENCES posters(id) ON DELETE CASCADE
    )
    """
    )

    # create handwritten relation
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS handwrittens(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        poster_id INTEGER,
        is_primary INTEGER,
        tlx INTEGER UNSIGNED NOT NULL,
        tly INTEGER UNSIGNED NOT NULL,
        trx INTEGER UNSIGNED NOT NULL,
        try INTEGER UNSIGNED NOT NULL,
        brx INTEGER UNSIGNED NOT NULL,
        bry INTEGER UNSIGNED NOT NULL,
        blx INTEGER UNSIGNED NOT NULL,
        bly INTEGER UNSIGNED NOT NULL,
        FOREIGN KEY (poster_id) REFERENCES posters(id) ON DELETE CASCADE,
        CONSTRAINT unique_poster_is_primary UNIQUE (poster_id, is_primary)
    )
    """
    )

    # create qrcodes relation
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS qrcodes_and_messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        poster_id INTEGER,
        qtlx INTEGER UNSIGNED,
        qtly INTEGER UNSIGNED,
        qtrx INTEGER UNSIGNED,
        qtry INTEGER UNSIGNED,
        qbrx INTEGER UNSIGNED,
        qbry INTEGER UNSIGNED,
        qblx INTEGER UNSIGNED,
        qbly INTEGER UNSIGNED,
        mtlx INTEGER UNSIGNED,
        mtly INTEGER UNSIGNED,
        mtrx INTEGER UNSIGNED,
        mtry INTEGER UNSIGNED,
        mbrx INTEGER UNSIGNED,
        mbry INTEGER UNSIGNED,
        mblx INTEGER UNSIGNED,
        mbly INTEGER UNSIGNED,
        FOREIGN KEY (poster_id) REFERENCES posters(id) ON DELETE CASCADE
    )
    """
    )


def close_connection():
    conn.commit()
    conn.close()


def upsert_poster(poster_image_name: str):
    cursor.execute(
        f"INSERT OR REPLACE INTO posters (image_name) VALUES ({poster_image_name})"
    )


def insert_all_new_posters(posters: dict):
    records = [(key, value) for key, values in posters.items() for value in values]
    insert_query = "INSERT OR IGNORE INTO posters (tree_type, image_name) VALUES (?, ?)"
    cursor.executemany(insert_query, records)
    conn.commit()


def clean_posters(posters: dict):
    cursor.execute("SELECT * FROM posters")
    fetched_posters = cursor.fetchall()
    records = [(key, value) for key, values in posters.items() for value in values]
    to_be_deleted_ids = []
    for item in fetched_posters:
        if tuple(item[-2:]) not in records:
            to_be_deleted_ids.append((item[0],))
    cursor.executemany(
        "DELETE FROM posters WHERE id = ?",
        to_be_deleted_ids,
    )
    conn.commit()


def get_poster_by_id(poster_id: int):
    cursor.execute("SELECT * FROM posters WHERE id = ?", (poster_id,))
    poster = cursor.fetchone()
    return poster


# def get_poster_by_name(poster_image_name: str):
#     cursor.execute("SELECT * FROM posters WHERE image_name = ?", (poster_image_name,))
#     poster = cursor.fetchone()
#     return poster


def get_posters_join_on_positions():
    query = """
    SELECT posters.id, posters.tree_type, posters.image_name, gift_takers.poster_id 
    FROM posters
    LEFT JOIN gift_takers ON posters.id=gift_takers.poster_id;
    """
    cursor.execute(query)
    gift_takers_records = cursor.fetchall()
    posters = {
        poster_id: {
            "tree_type": tree_type,
            "image_name": image_name,
            "has_gift_taker_pos": gift_pos_id is not None,
            "has_info_box_pos": False,
            "has_primary_handwritten_pos": False,
            "has_secondary_handwritten_pos": False,
            "only_message_pos": False,
            "only_qrcode_pos": False,
            "qrcode_and_message_pos": False,
        }
        for poster_id, tree_type, image_name, gift_pos_id in gift_takers_records
    }

    query = """
    SELECT posters.id, information_boxes.poster_id
    FROM posters
    LEFT JOIN information_boxes ON posters.id=information_boxes.poster_id;
    """
    cursor.execute(query)
    information_boxes_records = cursor.fetchall()
    for poster_id, info_box_id in information_boxes_records:
        posters[poster_id]["has_info_box_pos"] = info_box_id is not None

    query = """
    SELECT posters.id, handwrittens.is_primary, handwrittens.id
    FROM posters
    LEFT JOIN handwrittens ON posters.id=handwrittens.poster_id;
    """
    cursor.execute(query)
    handwrittens_records = cursor.fetchall()

    for poster_id, is_primary, handwritten_id in handwrittens_records:
        if handwritten_id:
            if is_primary:
                posters[poster_id]["has_primary_handwritten_pos"] = True
            else:
                posters[poster_id]["has_secondary_handwritten_pos"] = True

    query = """
    SELECT posters.id, qrcodes_and_messages.qtlx ,qrcodes_and_messages.mtlx, qrcodes_and_messages.poster_id
    FROM posters
    LEFT JOIN qrcodes_and_messages ON posters.id=qrcodes_and_messages.poster_id;
    """
    cursor.execute(query)
    qrcodes_messages_records = cursor.fetchall()
    for poster_id, qtlx, mtlx, qrcode_message_id in qrcodes_messages_records:
        if qrcode_message_id:
            if qtlx and not mtlx:
                posters[poster_id]["only_qrcode_pos"] = True
            elif mtlx and not qtlx:
                posters[poster_id]["only_message_pos"] = True
            elif mtlx and qtlx:
                posters[poster_id]["qrcode_and_message_pos"] = True
        else:
            posters[poster_id]["only_message_pos"] = False
            posters[poster_id]["only_qrcode_pos"] = False
            posters[poster_id]["qrcode_and_message_pos"] = False

    return posters


def delete_poster(poster_image_name: str):
    cursor.execute(f"DELETE FROM posters WHERE image_name = {poster_image_name}")
    conn.commit()


def upsert_gift_taker_pos(poster_id, positions: np.ndarray):
    tlx_, tly_ = positions[0]
    trx_, try_ = positions[1]
    brx_, bry_ = positions[2]
    blx_, bly_ = positions[3]

    upsert_query = """
    INSERT OR REPLACE INTO gift_takers (poster_id, tlx, tly, trx, try, brx, bry, blx, bly)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor.execute(
        upsert_query, (poster_id, tlx_, tly_, trx_, try_, brx_, bry_, blx_, bly_)
    )

    conn.commit()


def upsert_information_box_pos(poster_id, positions: np.ndarray):
    tlx_, tly_ = positions[0]
    trx_, try_ = positions[1]
    brx_, bry_ = positions[2]
    blx_, bly_ = positions[3]

    upsert_query = """
    INSERT OR REPLACE INTO information_boxes (poster_id, tlx, tly, trx, try, brx, bry, blx, bly)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor.execute(
        upsert_query, (poster_id, tlx_, tly_, trx_, try_, brx_, bry_, blx_, bly_)
    )

    conn.commit()


def upsert_handwritten_pos(poster_id, is_primary: bool, positions: np.ndarray):
    tlx_, tly_ = positions[0]
    trx_, try_ = positions[1]
    brx_, bry_ = positions[2]
    blx_, bly_ = positions[3]

    is_primary_bit = 1 if is_primary else 0

    upsert_query = """
    INSERT OR REPLACE INTO handwrittens (poster_id, is_primary, tlx, tly, trx, try, brx, bry, blx, bly)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor.execute(
        upsert_query,
        (poster_id, is_primary_bit, tlx_, tly_, trx_, try_, brx_, bry_, blx_, bly_),
    )

    conn.commit()


# TODO : must have a way to be done better (maybe using only sql and not in pythonic way)
def upsert_qrcode_message_pos(
    poster_id, qrcode_positions: np.ndarray, message_positions: np.ndarray
):
    if not np.all(qrcode_positions) and not np.all(message_positions):
        raise ValueError(
            "at least on of qrcode_positions or message_positions must have value."
        )

    if np.all(qrcode_positions):
        qtlx_, qtly_ = qrcode_positions[0]
        qtrx_, qtry_ = qrcode_positions[1]
        qbrx_, qbry_ = qrcode_positions[2]
        qblx_, qbly_ = qrcode_positions[3]
    else:
        qtlx_, qtly_ = None, None
        qtrx_, qtry_ = None, None
        qbrx_, qbry_ = None, None
        qblx_, qbly_ = None, None

    if np.all(message_positions):
        mtlx_, mtly_ = message_positions[0]
        mtrx_, mtry_ = message_positions[1]
        mbrx_, mbry_ = message_positions[2]
        mblx_, mbly_ = message_positions[3]
    else:
        mtlx_, mtly_ = None, None
        mtrx_, mtry_ = None, None
        mbrx_, mbry_ = None, None
        mblx_, mbly_ = None, None

    if qtlx_ and mtlx_:
        query = """
        SELECT id FROM qrcodes_and_messages WHERE poster_id = ? AND qtlx IS NOT NULL AND mtlx IS NOT NULL;
        """
        cursor.execute(query, (poster_id,))
        record = cursor.fetchone()
        record_id = record[0] if record else None

    elif qtlx_ and not mtlx_:
        query = """
        SELECT id FROM qrcodes_and_messages WHERE poster_id = ? AND qtlx IS NOT NULL AND mtlx IS  NULL;
        """
        cursor.execute(query, (poster_id,))
        record = cursor.fetchone()
        record_id = record[0] if record else None

    elif not qtlx_ and mtlx_:
        query = """
        SELECT id FROM qrcodes_and_messages WHERE poster_id = ? AND qtlx IS NULL AND mtlx IS NOT NULL;
        """
        cursor.execute(query, (poster_id,))
        record = cursor.fetchone()
        record_id = record[0] if record else None

    # print(record_id)

    upsert_query = """
    INSERT OR REPLACE INTO qrcodes_and_messages (id, poster_id, qtlx, qtly, qtrx, qtry, qbrx, qbry, qblx, qbly, mtlx, mtly, mtrx, mtry, mbrx, mbry, mblx, mbly)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor.execute(
        upsert_query,
        (
            record_id,
            poster_id,
            qtlx_,
            qtly_,
            qtrx_,
            qtry_,
            qbrx_,
            qbry_,
            qblx_,
            qbly_,
            mtlx_,
            mtly_,
            mtrx_,
            mtry_,
            mbrx_,
            mbry_,
            mblx_,
            mbly_,
        ),
    )

    conn.commit()


def _blob_coordinates_tuple_to_ndarray(coordinates):
    coordinates = [int.from_bytes(b, byteorder="little") if b else None for b in coordinates]
    roi_array = np.array(coordinates).reshape(-1, 2)
    return roi_array


# def get_gift_taker_pos(poster_id: int):
#     query = "SELECT * FROM gift_takers WHERE poster_id = ?"
#     cursor.execute(query, (poster_id,))
#     gift_taker_pos = cursor.fetchone()

#     if gift_taker_pos:
#         roi_array = _blob_coordinates_tuple_to_ndarray(gift_taker_pos[1:])
#         return gift_taker_pos[0], roi_array

#     return None, None


# def get_information_box_pos(poster_id: int):
#     query = "SELECT * FROM information_boxes WHERE poster_id = ?"
#     cursor.execute(query, (poster_id,))
#     information_box_pos = cursor.fetchone()

#     if information_box_pos:
#         roi_array = _blob_coordinates_tuple_to_ndarray(information_box_pos[1:])
#         return information_box_pos[0], roi_array

#     return None, None


# def get_handwrittens_pos(poster_id: int):
#     query = "SELECT * FROM handwrittens WHERE poster_id = ?"
#     cursor.execute(query, (poster_id,))
#     handwrittens_pos = cursor.fetchall()

#     if handwrittens_pos:
#         positions = []
#         for position in handwrittens_pos:
#             roi_array = _blob_coordinates_tuple_to_ndarray(position[3:])
#             positions.append((position[0], position[2], roi_array))

#         return poster_id, positions

#     return None, None


# def get_qrcodes_and_messages_pos(poster_id: int):
#     query = "SELECT * FROM qrcodes_and_messages WHERE poster_id = ?"
#     cursor.execute(query, (poster_id,))
#     qrcodes_and_messages_pos = cursor.fetchall()

#     if qrcodes_and_messages_pos:
#         positions = []
#         for position in qrcodes_and_messages_pos:
#             q_roi_array = _blob_coordinates_tuple_to_ndarray(position[2:10])
#             m_roi_array = _blob_coordinates_tuple_to_ndarray(position[10:])
#             positions.append((position[0], position[2], q_roi_array, m_roi_array))

#         return poster_id, positions

#     return None, None


def get_all_posters_data():
    # queries
    posters_query = "SELECT * FROM posters;"
    gift_takers_query = "SELECT * FROM gift_takers;"
    information_boxes_query = "SELECT * FROM information_boxes;"
    handwrittens_query = "SELECT * FROM handwrittens;"
    qrcodes_and_messages_query = "SELECT * FROM qrcodes_and_messages;"

    # fetch data
    posters = cursor.execute(posters_query).fetchall()
    gift_takers = cursor.execute(gift_takers_query).fetchall()
    information_boxes = cursor.execute(information_boxes_query).fetchall()
    handwrittens = cursor.execute(handwrittens_query).fetchall()
    qrcodes_and_messages = cursor.execute(qrcodes_and_messages_query).fetchall()

    # conver blob to int
    gift_takers = [
        (item[0], _blob_coordinates_tuple_to_ndarray(item[1:])) for item in gift_takers
    ]
    information_boxes = [
        (item[0], _blob_coordinates_tuple_to_ndarray(item[1:]))
        for item in information_boxes
    ]
    handwrittens = [
        (*item[1:3], _blob_coordinates_tuple_to_ndarray(item[3:]))
        for item in handwrittens
    ]
    qrcodes_and_messages = [
        (*item[1:2], _blob_coordinates_tuple_to_ndarray(item[2:10]), _blob_coordinates_tuple_to_ndarray(item[10:]))
        for item in qrcodes_and_messages
    ]

    return posters, gift_takers, information_boxes, handwrittens, qrcodes_and_messages

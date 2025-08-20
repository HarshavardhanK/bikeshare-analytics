#Generates parameterized SQL queries from user intent and mapped columns
import re

class SQLGenerator:
    def __init__(self, schema):
        self.schema = schema

    def generate(self, intent, mappings):
        select = intent.get('select')
        where = intent.get('where', [])
        group_by = intent.get('group_by')
        join = intent.get('join', [])
        agg = intent.get('agg')
        params = intent.get('params', [])
        order_by = intent.get('order_by')
        limit = intent.get('limit')

        # If select is a list, join with commas
        if isinstance(select, list):
            select_str = ', '.join(select)
        else:
            select_str = select

        # TIMESTAMPDIFF(minute, started_at, ended_at) -> EXTRACT(EPOCH FROM ended_at - started_at)/60.0
        if select_str and re.match(r"TIMESTAMPDIFF\s*\(\s*minute\s*,\s*started_at\s*,\s*ended_at\s*\)", select_str, re.IGNORECASE):
            select_str = "EXTRACT(EPOCH FROM ended_at - started_at)/60.0"
        # If select is an interval expression, wrap with EXTRACT(EPOCH FROM ...)/60.0 for minutes
        elif select_str and re.search(r'\bended_at\b\s*-\s*\bstarted_at\b', select_str) and 'EXTRACT' not in select_str:
            select_str = f"EXTRACT(EPOCH FROM {select_str})/60.0"

        # SELECT clause
        if agg and not group_by:
            select_clause = f"SELECT {agg}({select_str})"
        elif agg and group_by:
            select_clause = f"SELECT {select_str}, {agg}({select_str})"
        else:
            select_clause = f"SELECT {select_str}"

        from_clause = f"FROM {intent.get('from')}"

        join_clause = ""
        if join:
            for j in join:
                if isinstance(j, dict):
                    join_clause += f" JOIN {j['table']} ON {j['on']}"
                elif isinstance(j, str):
                    join_clause += f" JOIN {j}"

        where_clause = ""
        conds = []
        values = []
        if where:
            for w in where:
                # Robust gender matching
                if w['col'].lower() == 'rider_gender':
                    conds.append("LOWER(rider_gender) = LOWER(%s)")
                    values.append(w['val'])
                # BETWEEN handling
                elif w['op'].lower() == 'between' and isinstance(w['val'], (list, tuple)) and len(w['val']) == 2:
                    conds.append(f"{w['col']} BETWEEN %s AND %s")
                    values.extend(w['val'])
                # IS NOT NULL/IS NOT
                elif w['op'].upper() in ('IS NOT NULL', 'IS NOT'):
                    conds.append(f"{w['col']} {w['op']}")
                else:
                    conds.append(f"{w['col']} {w['op']} %s")
                    values.append(w['val'])
            where_clause = " WHERE " + " AND ".join(conds)

        # If group_by is a list, join with commas
        if isinstance(group_by, list):
            group_by_clause = f" GROUP BY {', '.join(group_by)}"
        elif group_by:
            group_by_clause = f" GROUP BY {group_by}"
        else:
            group_by_clause = ""

        # If order_by is a list, join with commas
        if isinstance(order_by, list):
            order_by_clause = f" ORDER BY {', '.join(order_by)}"
        elif order_by and order_by.startswith("COUNT(["):  # Fix COUNT(['col1', 'col2'])
            # Extract first column from list
            m = re.match(r"COUNT\(\[([^\]]+)\]\)", order_by)
            if m:
                first_col = m.group(1).split(',')[0].replace("'", "").strip()
                order_by_clause = f" ORDER BY COUNT({first_col}) DESC"
            else:
                order_by_clause = f" ORDER BY {order_by}"
        else:
            order_by_clause = f" ORDER BY {order_by}" if order_by else ""

        limit_clause = f" LIMIT {limit}" if limit else ""

        sql = f"{select_clause} {from_clause}{join_clause}{where_clause}{group_by_clause}{order_by_clause}{limit_clause};"
        return sql.strip(), values

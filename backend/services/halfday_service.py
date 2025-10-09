from typing import List, Dict, Tuple, Optional


def debug_log(message: str, category: str = "general"):

    try:
        from backend.config.settings import Config
        if category == "odoo_data" and Config.DEBUG_ODOO_DATA:
            print(f"DEBUG: {message}")
        elif category == "bot_logic" and Config.DEBUG_BOT_LOGIC:
            print(f"DEBUG: {message}")
        elif category == "knowledge_base" and Config.DEBUG_KNOWLEDGE_BASE:
            print(f"DEBUG: {message}")
        elif category == "general" and Config.VERBOSE_LOGS:
            print(f"DEBUG: {message}")
    except Exception:
        pass


class HalfDayLeaveService:
    """Encapsulates Half Day (custom hours) behavior for time-off flow.

    Responsibilities:
    - Replace the 'Unpaid Leave' option with a synthetic 'Half Days' option.
    - Map 'Half Days' to the base 'Annual Leave' type for submission.
    - Provide extra fields required to tick Odoo's Custom Hours (request_unit_hours).
    """

    HALF_DAY_NAME = "Custom Hours"

    def __init__(self):
        pass

    def _find_annual_leave_id(self, leave_types: List[Dict]) -> Optional[int]:
        for lt in leave_types:
            try:
                if isinstance(lt, dict) and lt.get('name') == 'Annual Leave':
                    return lt.get('id')
            except Exception:
                continue
        return None

    def replace_unpaid_with_halfdays(self, leave_types: List[Dict]) -> List[Dict]:
        """Return a new list where 'Unpaid Leave' is replaced by a synthetic 'Half Days' option.

        The synthetic option includes a marker 'special_code' and 'base_leave_type_id' pointing to Annual Leave.
        If Annual Leave is not found, this will no-op and simply remove Unpaid Leave.
        """
        if not leave_types or not isinstance(leave_types, list):
            return leave_types

        annual_id = self._find_annual_leave_id(leave_types)
        debug_log(f"HalfDay: Resolved Annual Leave ID: {annual_id}", "bot_logic")

        new_list: List[Dict] = []
        unpaid_replaced = False

        for lt in leave_types:
            name = (lt.get('name') or '').strip()
            if name == 'Unpaid Leave':
                if annual_id is not None:
                    half_day_entry = {
                        'id': f"halfday_{annual_id}",
                        'name': self.HALF_DAY_NAME,
                        'active': True,
                        'special_code': 'halfday',
                        'base_leave_type_id': annual_id
                    }
                    new_list.append(half_day_entry)
                    unpaid_replaced = True
                # If we can't resolve annual, we drop Unpaid Leave silently
            else:
                new_list.append(lt)

        # If there was no explicit 'Unpaid Leave' in the data, we still add Half Days at the end (if annual exists)
        if not unpaid_replaced and annual_id is not None:
            debug_log("HalfDay: 'Unpaid Leave' not found; appending 'Half Days' explicitly", "bot_logic")
            new_list.append({
                'id': f"halfday_{annual_id}",
                'name': self.HALF_DAY_NAME,
                'active': True,
                'special_code': 'halfday',
                'base_leave_type_id': annual_id
            })

        return new_list

    def is_halfday(self, selected_type: Dict) -> bool:
        if not isinstance(selected_type, dict):
            return False
        name = (selected_type.get('name') or '').strip()
        if name == self.HALF_DAY_NAME:
            return True
        return selected_type.get('special_code') == 'halfday'

    def build_submission(self, selected_type: Dict) -> Tuple[Optional[int], Dict]:
        """Return (leave_type_id, extra_fields) for submission.

        - For Half Days: maps to base Annual Leave ID and adds request_unit_hours: True.
        - For others: returns the selected id and empty extra fields.
        """
        if self.is_halfday(selected_type):
            base_id = selected_type.get('base_leave_type_id')
            extra_fields = {
                # Tick Odoo's Custom Hours toggle to enable hour fields
                'request_unit_hours': True
            }
            return base_id, extra_fields

        return selected_type.get('id'), {}



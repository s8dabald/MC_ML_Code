"""
Hilfsfunktionen fuer Skill-Aktionen.
Direkte mineflayer-APIs — keine Fire-and-forget-Bridges noetig,
da Free-Running normal auf Promises antwortet.
"""


def find_food_slot(bot):
    """Findet den Slot mit Essen im Inventar. Gibt Slot-Index zurueck oder None."""
    FOOD_IDS = {
        364, 320, 423, 366, 412,  # cooked meats
        297, 322, 260, 391, 393,  # bread, golden apple, apple, carrot, baked potato
        282, 360, 457,            # mushroom stew, melon, carrot
    }
    for i in range(9, 45):
        slot = bot.inventory.slots[i]
        if slot and slot.type in FOOD_IDS:
            return i
    return None


def find_best_tool_slot(bot, mcData):
    """Findet das beste Werkzeug/Schwert im Inventar. Gibt Slot-Index zurueck."""
    PRIORITY = [272, 268, 274, 270]  # stone_sword, wood_pick, stone_pick, wood_sword
    for tool_id in PRIORITY:
        for i in range(9, 45):
            slot = bot.inventory.slots[i]
            if slot and slot.type == tool_id:
                return i
    return None


def craft_item(bot, mcData, item_id):
    """
    Versucht ein Item zu craften.
    Nutzt bot.recipesFor() und bot.craft() — loest normal auf (Free-Running).
    """
    try:
        table = find_crafting_table(bot)

        if table is not None:
            recipes = bot.recipesFor(item_id, None, 1, table)
            if recipes is not None and recipes.length > 0:
                bot.craft(recipes[0], 1, table)
                return True

        # 2x2 Crafting (ohne Table)
        recipes = bot.recipesFor(item_id, None, 1, None)
        if recipes is not None and recipes.length > 0:
            bot.craft(recipes[0], 1, None)
            return True

        return False
    except Exception:
        return False


def find_crafting_table(bot, radius=2):
    """Naechsten Crafting-Table finden."""
    from javascript import require
    Vec3 = require("vec3")
    pos = bot.entity.position
    for dy in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                block = bot.blockAt(Vec3(
                    int(pos.x) + dx,
                    int(pos.y) + dy,
                    int(pos.z) + dz,
                ))
                if block and block.name == "crafting_table":
                    return block
    return None


def eat_food(bot):
    """
    Essen benutzt: findet Essen im Inventar, equippt es, und rechtsklickt.
    """
    slot = find_food_slot(bot)
    if slot is None:
        return False

    bot.clickWindow(slot, 0, 0)
    bot.clickWindow(36, 0, 0)
    bot.setQuickBarSlot(0)
    bot.activateItem(False)
    return True


def equip_best(bot, mcData):
    """Bestes Item equippen."""
    slot = find_best_tool_slot(bot, mcData)
    if slot is None:
        return False

    bot.clickWindow(slot, 0, 0)
    bot.clickWindow(36, 0, 0)
    bot.setQuickBarSlot(0)
    return True

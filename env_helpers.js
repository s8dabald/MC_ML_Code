// env_helpers.js — Schnelle Obs-Sammlung in einem einzigen Bridge-Call.
// Wird per require("./env_helpers.js") aus Python geladen.
// Gibt ein Float64Array der Laenge 99 zurueck (statt ~200 Einzel-Calls).

// Globale Promise-Rejections abfangen, damit eine Promise, die nach einem
// Disconnect/Kick auf einen toten Bot verweist, die Node-Session NICHT crasht
// (sonst: "[JSE] triggerUncaughtException ... PyBridge.call ... failed").
process.on('unhandledRejection', (reason) => {
  try {
    console.log('[JSE][unhandledRejection]', String((reason && reason.message) || reason))
  } catch (_) {}
})
process.on('uncaughtException', (err) => {
  try {
    console.log('[JSE][uncaughtException]', String((err && err.message) || err))
  } catch (_) {}
})

// Mappt mineflayer-Entity-Typ-Strings ('animal', 'mob', 'hostile', 'player', ...)
// auf stabile Zahlen, damit das Obs rein numerisch ist (float32-tauglich). Der
// rohe String wuerde np.array(list(proxy)) mit "could not convert string" brechen.
function entityTypeNum(t) {
  if (typeof t !== 'string') return 0
  var classified = {
    'player': 1,
    'hostile': 2,
    'mob_hostile': 2,
    'mob': 3,
    'animal': 4,
    'mob_neutral': 5,
    'ambient': 6,
    'water_creature': 7,
    'water_ambient': 8,
    'water_animal': 9,
    'villager': 10,
    'other': 11,
    'object': 12,
    'boat': 13,
    'dropped_item': 14,
    'arrow': 15,
    'thrown_entity': 16,
    'xporb': 17,
    'explosive': 18,
  }
  return Object.prototype.hasOwnProperty.call(classified, t) ? classified[t] : 0
}

module.exports = {
  // Baut das Obs-Array (99 Features) in einem Bridge-Call. Intern von snapshot
  // und step genutzt, damit die Obs-Erzeugung nur EINMAL im Code steht.
  _buildObs: function (bot) {
    if (!bot || !bot.entity) return new Array(99).fill(0)

    // Voller Feld-Aufbau in try/catch: Wenn der Bot mitten im Step stirbt/respawnt
    // (Race zwischen dem obigen Guard und dem Zugriff auf bot.entity.position o.ae.),
    // werfen diese Zugriffe eine JS-Exception. Damals brach das den ganzen Schritt-
    // Call und lieferte ein verkuerztes serialize-Ergebnis -> IndexError in Python.
    // Hier: bei jeder Exception deterministisch ein 99er-Nullen-Array zurueckgeben.
    try {
      var pos = bot.entity.position
      var out = []
      var i = 0

      // Player (5)
      out[i++] = pos.x
      out[i++] = pos.y
      out[i++] = pos.z
      out[i++] = bot.entity.yaw || 0
      out[i++] = bot.entity.pitch || 0

      // Status (4)
      out[i++] = bot.health != null ? bot.health : 20
      out[i++] = bot.food != null ? bot.food : 20
      out[i++] = bot.foodSaturation != null ? bot.foodSaturation : 5
      out[i++] = bot.oxygenLevel != null ? bot.oxygenLevel : 0

      // XP (3)
      out[i++] = bot.experience.level != null ? bot.experience.level : 0
      out[i++] = bot.experience.points != null ? bot.experience.points : 0
      out[i++] = bot.experience.progress != null ? bot.experience.progress : 0

      // Game (5)
      var gm = bot.game.gameMode
      var gmNum = 0
      if (gm === "creative") gmNum = 1
      else if (gm === "adventure") gmNum = 2
      else if (gm === "spectator") gmNum = 3

      var diff = bot.game.difficulty
      var diffNum = 2
      if (diff === "peaceful") diffNum = 0
      else if (diff === "easy") diffNum = 1
      else if (diff === "hard") diffNum = 3

      var dim = bot.game.dimension
      var dimNum = 0
      if (dim === "the_nether") dimNum = -1
      else if (dim === "the_end") dimNum = 1

      out[i++] = dimNum
      out[i++] = diffNum
      out[i++] = gmNum
      out[i++] = bot.game.minY != null ? bot.game.minY : -64
      out[i++] = bot.game.height != null ? bot.game.height : 384

      // Time (3)
      out[i++] = bot.time.timeOfDay != null ? bot.time.timeOfDay : 0
      out[i++] = bot.time.isDay != null ? (bot.time.isDay ? 1 : 0) : 1
      out[i++] = bot.time.day != null ? bot.time.day : 0

      // Weather (3)
      out[i++] = bot.isRaining ? 1 : 0
      out[i++] = bot.rainState != null ? bot.rainState : 0
      out[i++] = bot.thunderState != null ? bot.thunderState : 0

      // Inventory (45)
      out[i++] = bot.currentWindow ? 1 : 0

      var held = bot.heldItem
      out[i++] = held ? held.type : 0
      out[i++] = held ? held.count : 0

      out[i++] = bot.quickBarSlot || 0

      for (var s = 36; s <= 44; s++) {
        var item = bot.inventory.slots[s]
        out[i++] = item ? item.type : 0
      }

      for (var s = 9; s <= 35; s++) {
        var item = bot.inventory.slots[s]
        out[i++] = item ? item.type : 0
      }

      for (var s = 5; s <= 8; s++) {
        var item = bot.inventory.slots[s]
        out[i++] = item ? item.type : 0
      }

      var offItem = bot.inventory.slots[45]
      out[i++] = offItem ? offItem.type : 0

      // Blocks (27, radius=1) — nutzt Vec3-Instanz des Bots
      var base = pos.floored()
      for (var dy = -1; dy <= 1; dy++) {
        for (var dz = -1; dz <= 1; dz++) {
          for (var dx = -1; dx <= 1; dx++) {
            var block = bot.blockAt(base.offset(dx, dy, dz))
            if (!block) {
              out[i++] = 0
            } else if (block.name === "air" || block.name === "cave_air") {
              out[i++] = 0
            } else {
              out[i++] = block.type
            }
          }
        }
      }

      // Nearest entity (4)
      try {
        var nearest = bot.nearestEntity()
        if (nearest && nearest !== bot.entity && nearest.position) {
          out[i++] = entityTypeNum(nearest.type) // type ist ein String ('animal' etc.) -> Zahl mappen
          out[i++] = nearest.position.x - pos.x
          out[i++] = nearest.position.y - pos.y
          out[i++] = nearest.position.z - pos.z
        } else {
          i += 4
        }
      } catch (e) {
        i += 4
      }

      return out
    } catch (e) {
      return new Array(99).fill(0)
    }
  },

  // Reine Obs-Sammlung (fuer eat/craft-Fallback und Kompatibilitaet).
  snapshot: function (bot) {
    return module.exports._buildObs(bot)
  },

  // BUNDLED STEP: fuehrt die haeufige Aktion (0-26) aus UND liefert in EINEM
  // Bridge-Call alles, was env.step() sonst in 4-6 separaten Roundtrips holt:
  //   Obs (99), health, lagTick (bot.time.age), alive, connected.
  // Rueckgabe: flaches Top-Level-Array
  //   [obs0..98, health, lagTick, alive(0/1), connected(0/1)]
  // Python konvertiert via res.valueOf() (1 IPC / serialize) -> echtes Python-list,
  // dann obs=vals[:99], health=vals[99], ...
  //
  // Aktionen 27 (eat) und 28 (craft) bleiben in Python (brauchen Helper dort).
  // `anyControlActive`: true wenn ein Bewegungs-State aktiv ist -> clearControlStates
  // nur dann, statt bei jeder Nicht-Bewegungs-Aktion (spart RPC bei idle).
  step: function (bot, actionType, mappedValue, rotation, anyControlActive) {
    if (bot && bot.entity) {
      var b = bot

      // Nicht-Bewegungs-Aktionen: Controls nur zuruecksetzen, wenn ein
      // Bewegungs-State wirklich aktiv ist.
      if (actionType < 1 || actionType > 7) {
        if (anyControlActive) {
          try { b.clearControlStates() } catch (e) {}
        }
      }

      if (actionType === 1) {
        b.setControlState("forward", true)
      } else if (actionType === 2) {
        b.setControlState("back", true)
      } else if (actionType === 3) {
        b.setControlState("left", true)
      } else if (actionType === 4) {
        b.setControlState("right", true)
      } else if (actionType === 5) {
        b.setControlState("jump", true)
      } else if (actionType === 6) {
        b.setControlState("sneak", true)
      } else if (actionType === 7) {
        b.setControlState("sprint", true)
      } else if (actionType === 8) {
        try { b.look(b.entity.yaw + rotation, b.entity.pitch, true) } catch (e) {}
      } else if (actionType === 9) {
        try { b.look(b.entity.yaw - rotation, b.entity.pitch, true) } catch (e) {}
      } else if (actionType === 10) {
        try { b.look(b.entity.yaw, b.entity.pitch + rotation, true) } catch (e) {}
      } else if (actionType === 11) {
        try { b.look(b.entity.yaw, b.entity.pitch - rotation, true) } catch (e) {}
      } else if (actionType === 12) {
        module.exports.attackNearest(b)
      } else if (actionType === 13) {
        module.exports.useBlock(b) // fire-and-forget
      } else if (actionType === 14) {
        try {
          var dblk = b.blockAtCursor(5)
          if (dblk && b.canDigBlock(dblk)) b.dig(dblk, "ignore")
        } catch (e) {}
      } else if (actionType === 15) {
        try {
          var pblk = b.blockAtCursor(5)
          if (pblk && b.heldItem) {
            b._placeBlockWithOptions(pblk, { x: 0, y: 1, z: 0 }, {
              "forceLook": "ignore",
              "swingArm": "right",
            })
          }
        } catch (e) {}
      } else if (actionType >= 16 && actionType <= 24) {
        try { b.setQuickBarSlot(actionType - 16) } catch (e) {}
      } else if (actionType === 25) {
        try { b.simpleClick.leftMouse(45) } catch (e) {}
      } else if (actionType === 26) {
        try { if (b.heldItem) b.tossStack(b.heldItem) } catch (e) {}
      }
      // actionType 0 (idle) und >26: nichts weiter im Bündel
    }

    var out = module.exports._buildObs(bot)  // 99 Felder (garantiert numerisch, siehe _buildObs)
    out.push(bot && bot.entity ? (bot.health != null ? bot.health : 0) : 0)   // 99 health
    out.push(bot && bot.time && bot.time.age != null ? bot.time.age : 0)      // 100 lagTick
    out.push(bot && bot.isAlive !== false ? 1 : 0)                            // 101 alive
    out.push(bot ? 1 : 0)                                                     // 102 connected

    // Hard-Safety: Das Bundle MUSS exakt 103 numerische Werte liefern, sonst crasht
    // Python beim res.valueOf()->[99] mit IndexError. Falls _buildObs aus irgendeinem
    // Grund (Race/Partiellfill) nicht 99 lieferte, auf exakt 103 mit Nullen auffuellen
    // bzw. abschneiden. So ist die Laenge deterministisch.
    var targetLen = 103
    if (out.length < targetLen) {
      while (out.length < targetLen) out.push(0)
    } else if (out.length > targetLen) {
      out.length = targetLen
    }
    return out
  },

  // use/rechtsklick: Block wird per Fadenkreuz-Raycast geholt. Wenn nichts im
  // Fadenkreuz ist, passiert nichts (Modell hat "Pech"). Sonst: force-look auf den
  // Block, damit der interne lookAt(..., false) von activateBlock keine Rotation
  // mehr braucht (deltaYaw ~ 0 -> sofortiger return, kein hang).
  useBlock: function (bot) {
    var block = bot.blockAtCursor(5)
    if (!block) return Promise.resolve(false)
    var target = block.position.offset(0.5, 0.5, 0.5)
    return new Promise(function (resolve, reject) {
      bot.lookAt(target, true)
        .then(function () { return bot.activateBlock(block) })
        .then(function () { resolve(true) }, reject)
    })
  },

  // attack auf naechstes, GARANTIERT noch verfuegbares Entity.
  // Root-Cause-Fix: bot.nearestEntity() kann eine Entity-Referenz liefern, die
  // dem Server schon als "invalid" bekannt ist (z.B. nach Tod/Distanz-Update).
  // Senden wir trotzdem use_entity mit toter ID, kickt Paper den Bot mit
  // "Attempting to attack an invalid entity" -> der eigentliche DC-Grund.
  // Hier pruefen wir die Entity nochmals gegen bot.entities (Client-Registry),
  // filtern den Bot selbst und blocken Angriffe auf tote/invalid Entities.
  attackNearest: function (bot) {
    if (!bot || !bot.entity) return false
    var ownId = bot.entity.id
    var best = null
    var bestDist = Infinity
    var ownPos = bot.entity.position
    for (var id in bot.entities) {
      var e = bot.entities[id]
      if (!e || id === ownId || e === bot.entity) continue
      if (!e.position) continue
      // Nur Mobs/Tiere/Fiendliche angreifen; keine Players, Drop-Items (object) usw.
      var kind = e.type
      if (kind !== 'mob' && kind !== 'hostile' && kind !== 'neutral' && kind !== 'animals') continue
      var d = e.position.distanceTo(ownPos)
      if (d < bestDist) {
        bestDist = d
        best = e
      }
    }
    if (!best) return false
    try {
      bot.attack(best)
      return true
    } catch (e) {
      console.log('[serve][attack]', String((e && e.message) || e))
      return false
    }
  }
}

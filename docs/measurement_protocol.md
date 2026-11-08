# Measurement protocol

**Status: stub.** This will become the step-by-step checklist for a measurement session. Sections get filled in as the matching tasks are done. Nothing here has been tried on hardware yet.

## What a session measures (plain language)
The thermal camera reads the leaf's temperature and a dry reference card's temperature in the same picture. The other sensors read air temperature, humidity, pressure and light. A leaf that is losing water through its stomata is cooler than a dry card under the same light, and the size of that difference lets us work out how open the stomata are (g_s).

## Before the session
- TODO: check the Pi clock is right (the logger will refuse to start on a clearly wrong date).
- TODO: plant watering, acclimation time, and where each plant sits (randomized blocks).
- TODO: card setup: same tilt, distance and backdrop as the leaf (assumption A16).
- TODO: fan on low so airflow is forced, not buoyant (assumption A17).

## During the session
- TODO: start the logger and mark events (lamp on, measurement start).
- TODO: how long to wait for steady state.

## After the session
- TODO: copy the session folder off the Pi the same day. Never edit raw files.

## Related
- [assumptions_and_limitations.md](assumptions_and_limitations.md)

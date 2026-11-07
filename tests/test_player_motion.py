import math
import unittest

from game.actions import Direction
from game.config import DEFAULT_BRAKING, PHYSICS_DT
from game.player import PlayerState, step_idle_player


class DirectionTests(unittest.TestCase):
    def test_eight_unit_directions_and_zero_idle(self):
        self.assertEqual(len(Direction), 9)
        for direction in Direction:
            with self.subTest(direction=direction):
                expected_length = 0.0 if direction == Direction.IDLE else 1.0
                self.assertAlmostEqual(math.hypot(*direction.vector), expected_length)

    def test_y_down_direction_mapping(self):
        expected_signs = {
            Direction.IDLE: (0, 0),
            Direction.NORTH: (0, -1),
            Direction.NORTHEAST: (1, -1),
            Direction.EAST: (1, 0),
            Direction.SOUTHEAST: (1, 1),
            Direction.SOUTH: (0, 1),
            Direction.SOUTHWEST: (-1, 1),
            Direction.WEST: (-1, 0),
            Direction.NORTHWEST: (-1, -1),
        }
        for direction, signs in expected_signs.items():
            with self.subTest(direction=direction):
                actual = tuple(
                    0 if value == 0 else int(math.copysign(1, value)) for value in direction.vector
                )
                self.assertEqual(actual, signs)


class IdleBrakingTests(unittest.TestCase):
    def test_fixed_timestep_and_provisional_braking_setting(self):
        self.assertEqual(PHYSICS_DT, 1.0 / 60.0)
        self.assertEqual(DEFAULT_BRAKING, 40.0)

    def test_toy_braking_and_velocity_first_position_update(self):
        player = PlayerState(position=(10.0, 20.0), velocity=(6.0, 0.0))
        result = step_idle_player(player, braking=12.0)
        self.assertAlmostEqual(result.velocity[0], 5.8)
        self.assertEqual(result.velocity[1], 0.0)
        self.assertAlmostEqual(result.position[0], 10.0 + 5.8 * PHYSICS_DT)
        self.assertEqual(result.position[1], 20.0)

    def test_stopping_ticks_for_both_discussed_braking_values(self):
        for braking, stopping_ticks in ((12.0, 30), (40.0, 9)):
            with self.subTest(braking=braking):
                player = PlayerState(position=(0.0, 0.0), velocity=(6.0, 0.0))
                for _ in range(stopping_ticks - 1):
                    player = step_idle_player(player, braking=braking)
                    self.assertGreater(player.velocity[0], 0.0)
                player = step_idle_player(player, braking=braking)
                self.assertEqual(player.velocity, (0.0, 0.0))
                self.assertEqual(step_idle_player(player, braking=braking), player)

    def test_braking_preserves_velocity_direction(self):
        for velocity in ((3.0, 4.0), (-3.0, 4.0), (-3.0, -4.0), (3.0, -4.0)):
            with self.subTest(velocity=velocity):
                player = PlayerState(position=(0.0, 0.0), velocity=velocity)
                result = step_idle_player(player, braking=40.0)
                speed = math.hypot(*result.velocity)
                self.assertAlmostEqual(speed, 5.0 - 40.0 * PHYSICS_DT)
                for before, after in zip(velocity, result.velocity):
                    self.assertAlmostEqual(after / speed, before / 5.0)

    def test_small_speed_stops_without_reversing(self):
        player = PlayerState(position=(1.0, 2.0), velocity=(-0.01, 0.01))
        result = step_idle_player(player)
        self.assertEqual(result.velocity, (0.0, 0.0))
        self.assertEqual(result.position, player.position)

    def test_final_stopping_tick_uses_zero_updated_velocity(self):
        player = PlayerState(position=(0.0, 0.0), velocity=(1.0, 0.0))
        penultimate = step_idle_player(player, braking=40.0)
        self.assertGreater(penultimate.position[0], 0.0)
        self.assertGreater(penultimate.velocity[0], 0.0)
        stopped = step_idle_player(penultimate, braking=40.0)
        self.assertEqual(stopped.velocity, (0.0, 0.0))
        self.assertEqual(stopped.position, penultimate.position)

    def test_stationary_player_stays_stationary(self):
        player = PlayerState(position=(1.0, 2.0))
        self.assertEqual(step_idle_player(player), player)

    def test_facing_stamina_and_original_state_are_preserved(self):
        player = PlayerState(
            position=(1.0, 2.0), velocity=(6.0, 0.0), facing=-math.pi / 2, stamina=37.0
        )
        result = step_idle_player(player)
        self.assertEqual(result.facing, player.facing)
        self.assertEqual(result.stamina, 37.0)
        self.assertEqual(player.position, (1.0, 2.0))
        self.assertEqual(player.velocity, (6.0, 0.0))
        self.assertIsNot(result, player)

    def test_zero_braking_preserves_coasting_velocity(self):
        player = PlayerState(position=(0.0, 0.0), velocity=(6.0, -3.0))
        result = step_idle_player(player, braking=0.0)
        self.assertEqual(result.velocity, player.velocity)
        self.assertAlmostEqual(result.position[0], 0.1)
        self.assertAlmostEqual(result.position[1], -0.05)

    def test_tiny_velocity_without_braking_is_not_snapped_to_zero(self):
        player = PlayerState(position=(0.0, 0.0), velocity=(1e-15, 0.0))
        result = step_idle_player(player, braking=0.0)
        self.assertEqual(result.velocity, player.velocity)
        self.assertGreater(result.position[0], 0.0)

    def test_replaying_same_initial_state_is_deterministic(self):
        initial = PlayerState(position=(2.0, 3.0), velocity=(3.0, 4.0), facing=0.7)

        def trajectory():
            player = initial
            states = []
            for _ in range(60):
                player = step_idle_player(player)
                states.append(player)
            return states

        self.assertEqual(trajectory(), trajectory())

    def test_invalid_braking_is_rejected(self):
        player = PlayerState(position=(0.0, 0.0))
        for braking in (-1.0, math.inf, -math.inf, math.nan):
            with self.subTest(braking=braking), self.assertRaises(ValueError):
                step_idle_player(player, braking=braking)

    def test_invalid_state_is_rejected(self):
        invalid_fields = (
            {"position": (math.nan, 0.0)},
            {"position": (0.0,)},
            {"velocity": (math.inf, 0.0)},
            {"velocity": (0.0, 0.0, 0.0)},
            {"facing": math.nan},
            {"stamina": -1.0},
            {"stamina": 101.0},
            {"stamina": math.nan},
        )
        for fields in invalid_fields:
            with self.subTest(fields=fields):
                arguments = {"position": (0.0, 0.0), **fields}
                with self.assertRaises(ValueError):
                    PlayerState(**arguments)


if __name__ == "__main__":
    unittest.main()

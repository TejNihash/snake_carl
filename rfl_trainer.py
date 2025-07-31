import pygame

# Initialize Pygame
pygame.init()

# Create a simple Sprite class (for demonstration)
class MySprite(pygame.sprite.Sprite):
    def __init__(self, color, x, y):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(color)
        self.rect = self.image.get_rect(topleft=(x, y))

# Create a sprite group
all_sprites = pygame.sprite.Group()

# Add some sprites to the group
sprite1 = MySprite((255, 0, 0), 100, 100)
sprite2 = MySprite((0, 255, 0), 200, 200)
sprite3 = MySprite((0, 0, 255), 300, 300)

all_sprites.add(sprite1, sprite2, sprite3)

# Get the first sprite from the group
first_sprite = all_sprites.sprites()[0]

print(f"The first sprite is: {first_sprite.rect.x}")

prev_head_coords  = (all_sprites.sprites()[0].rect.x ,all_sprites.sprites()[0].rect.y)

all_sprites.sprites()[0].rect.x = 90

a = prev_head_coords

print("prev",prev_head_coords)
print("a",a)
print(first_sprite.rect.x)

#iterate through a sprite group or use sprites()?

for sprite_obj in all_sprites:
    print(sprite_obj.rect.x)


a = True

print(not a)
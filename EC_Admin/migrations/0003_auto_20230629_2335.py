# Generated by Django 2.2.3 on 2023-06-29 20:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('EC_Admin', '0002_auto_20230629_2329'),
    ]

    operations = [
        migrations.AlterField(
            model_name='constituency',
            name='constituency_type',
            field=models.CharField(choices=[('Parliamentary', 'Parliamentary'), ('Assembly', 'Assembly')], max_length=50),
        ),
    ]

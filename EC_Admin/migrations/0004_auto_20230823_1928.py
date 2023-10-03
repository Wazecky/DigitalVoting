# Generated by Django 2.2.3 on 2023-08-23 16:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('EC_Admin', '0003_auto_20230629_2335'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='voters',
            name='constituency',
        ),
        migrations.AddField(
            model_name='candidates',
            name='assembly',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='candidates',
            name='parliamentary',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='voters',
            name='assembly',
            field=models.CharField(default='Your Assembly', max_length=60),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='voters',
            name='parliamentary',
            field=models.CharField(default='Your Parliamentary', max_length=60),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='candidates',
            name='constituency',
            field=models.CharField(max_length=60),
        ),
        migrations.DeleteModel(
            name='Constituency',
        ),
    ]